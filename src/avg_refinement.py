"""AVG case assembly and reuse without mixing distinct priors or objectives."""
from fractions import Fraction as F
import json
from pathlib import Path
import time

import numpy as np

from .avg_prior import Prior, uniform_grid
from .avg_experiments import avg_upper, verify_result
from .avg_accelerated import (solve_avg_dual_accelerated, solve_avg_primal_robust,
                              transfer_patterns)
from .patterns import separating_patterns
from .exact import seal, outward, save_json


def prior_key(prior):
    """Names are labels; normalized masses, edges and model define the problem."""
    d = prior.as_dict() if isinstance(prior, Prior) else Prior(**prior).as_dict()
    return json.dumps({k: d[k] for k in ('model', 'edges', 'masses')}, sort_keys=True)


def load_seed(path, prior, epsilon):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_result(d)
    if prior_key(d['prior']) != prior_key(prior) or F(d['epsilon']) != F(str(epsilon)):
        raise ValueError('AVG seed is for a different prior/model/epsilon')
    return d


def seed_bits(path, seed, grid, role):
    if seed is None:
        return None
    npz = Path(path).parent/role/'patterns.npz'
    if npz.exists():
        return transfer_patterns(npz, grid)
    # Older result formats contain dual profiles but no persisted pattern pool.
    d = seed['lower']
    aa, bb, pi = [np.array([float(F(v)) for v in d[k]]) for k in ('alpha','beta','weights')]
    old = np.array([float(F(v)) for v in d['grid']])
    bits = separating_patterns(aa, bb, pi)
    x = np.asarray(grid, float)
    right = np.clip(np.searchsorted(old, x), 1, len(old)-1)
    near = np.where(abs(x-old[right-1]) <= abs(x-old[right]), right-1, right)
    return bits[:, near]


def run_case_accelerated(prior, epsilon, m, args, progress, folder):
    start = time.perf_counter()
    folder = Path(folder)
    partition = uniform_grid(m)
    lower_x, pi, D = prior.coarsen(partition)
    upper_x = lower_x if prior.model == 'discrete' else partition
    upper_x = [F(str(float(v))) for v in upper_x]
    weights, m2 = prior.hat_weights(upper_x), prior.moments()[2]
    seed_path = getattr(args, 'seed_result', None)
    seed = load_seed(seed_path, prior, epsilon) if seed_path else None
    common = dict(max_cuts=args.max_cuts, time_limit=args.solver_seconds,
                  pattern_mode=args.patterns, symmetry=not args.no_symmetry, progress=progress)
    def lower_checkpoint(cert):
        save_json(folder/'lower_context.json', dict(prior=prior.as_dict(), epsilon=str(epsilon),
            requested_m=m, partition=list(map(str, partition)), within_bin_variance=str(D), certificate=cert))
    progress(f'AVG lower-only dual: M={len(lower_x)}, within-bin correction D={float(D):.6g}')
    lower = solve_avg_dual_accelerated(epsilon, lower_x, pi, m2,
        seed_patterns=seed_bits(seed_path, seed, lower_x, 'lower'), folder=folder/'lower',
        checkpoint=lower_checkpoint, **common)

    def diagnostic(solution):
        return {k: solution[k] for k in ('search_complete','history','pattern_mode',
                                         'runtime_seconds','stop_reason','symmetry','dual_status')}

    def assemble(channel, upper_info, reused):
        U, L = avg_upper(prior, channel), F(lower['certificate']['L_rational'])
        d = seal(dict(kind='avg_result', epsilon=str(epsilon), prior=prior.as_dict(),
            requested_m=m, partition=list(map(str, partition)), within_bin_variance=str(D),
            channel=channel, lower=lower['certificate'], U_rational=str(U), U_cert=outward(U, True),
            gap_rational=str(U-L), gap_cert=outward(U-L, True), lower_bin_value_cert=outward(L+D,False),
            upper_solver=upper_info, lower_solver=diagnostic(lower),
            raw_channel_check={}, runtime_seconds=time.perf_counter()-start,
            acceleration=dict(seed_result=str(seed_path) if seed_path else None,
                              channel_reused=reused, dual_coefficients_transferred=False,
                              solver_warm_start=False, lower_primal_skipped=True),
            meaning="bounds on the given prior's average conditional variance, all x in [-1,1] unbiased and LDP"))
        verify_result(d)
        return d

    if seed:
        save_json(folder/'checkpoint_result.json', assemble(seed['channel'],
            dict(search_complete=False, status='previous_channel_lower_checkpoint'), True))
    progress(f'AVG upper weighted SOCP: M={len(upper_x)}')
    upper_info = dict(search_complete=False)
    try:
        if upper_x == lower_x and weights == pi:
            upper = lower
        else:
            upper = solve_avg_dual_accelerated(epsilon, upper_x, weights, m2,
                seed_patterns=seed_bits(seed_path, seed, upper_x, 'upper'), folder=folder/'upper', **common)
        upper_info = diagnostic(upper)
        channel, attempts = solve_avg_primal_robust(epsilon, upper_x, weights, prior, upper['bits'],
            upper['certificate']['L_rational'], time_limit=args.solver_seconds,
            symmetry=not args.no_symmetry, folder=folder, progress=progress)
        upper_info['primal_attempts'] = attempts
        reused = seed is not None and avg_upper(prior, seed['channel']) < avg_upper(prior, channel)
        if reused:
            channel = seed['channel']
    except (RuntimeError, ValueError) as exc:
        save_json(folder/'upper_error.json', dict(error_type=type(exc).__name__, error=str(exc)))
        if seed is None:
            raise
        progress(f'AVG upper failed; previous certified channel retained: {exc}')
        channel, reused = seed['channel'], True
        upper_info.update(status='failed_previous_channel_retained', error=str(exc))
    return assemble(channel, upper_info, reused)
