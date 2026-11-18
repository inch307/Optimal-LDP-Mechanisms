"""Numerical AVG experiments with independently verifiable prior-specific bounds."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from .avg_prior import Prior, uniform_grid
from .avg_solver import solve_avg, certify_avg_dual, verify_avg_dual
from .channel import from_moments, verify_grid_channel
from .certify_continuous import certify_continuous, verify_continuous, channel_from_certificate
from .exact import seal, check_hash, outward, save_json


def avg_upper(prior, channel):
    weights = prior.hat_weights(channel["grid"])
    y = [F(v) for v in channel["outputs"]]
    seconds = [sum(F(p)*a*a for p, a in zip(row, y)) for row in channel["P_grid"]]
    return sum(w*s for w, s in zip(weights, seconds)) - prior.moments()[2]


def verify_result(data):
    """Recheck the prior, coarsening, every dual profile, and the explicit channel."""
    check_hash(data)
    if data["kind"] != "avg_result":
        raise ValueError("wrong result type")
    prior = Prior(**data["prior"])
    channel, lower = data["channel"], data["lower"]
    if channel["epsilon"] != lower["epsilon"] or lower["epsilon"] != data["epsilon"]:
        raise ValueError("epsilon mismatch")
    verify_continuous(channel)
    verify_avg_dual(lower)
    x, pi, D = prior.coarsen(data["partition"])
    if (list(map(str, x)) != lower["grid"] or list(map(str, pi)) != lower["weights"]
            or str(prior.moments()[2]) != lower["second_moment"]
            or str(D) != data["within_bin_variance"]):
        raise ValueError("lower certificate is for a different prior/coarsening")
    U, L = avg_upper(prior, channel), F(lower["L_rational"])
    if (U < L or str(U) != data["U_rational"] or str(U-L) != data["gap_rational"]
            or F.from_float(data["U_cert"]) < U
            or F.from_float(data["gap_cert"]) < U-L):
        raise ValueError("invalid AVG upper bound/gap")
    return True


def run_case(prior, epsilon, m, args, progress):
    started = time.perf_counter()
    partition = uniform_grid(m)
    lower_x, pi, D = prior.coarsen(partition)
    # Discrete priors include all occupied bin centers and zero-weight endpoints.
    upper_x = lower_x if prior.model == "discrete" else partition
    # Channel serialization uses these exact decimal representations of float knots.
    upper_x = [F(str(float(v))) for v in upper_x]
    weights = prior.hat_weights(upper_x)
    m2 = prior.moments()[2]
    kwargs = dict(pattern_mode=args.patterns, max_cuts=args.max_cuts, time_limit=args.solver_seconds)
    progress(f"upper SOCP M={len(upper_x)}")
    upper_solution = solve_avg(epsilon, upper_x, weights, m2, progress=progress, **kwargs)
    raw = from_moments(epsilon, [float(v) for v in upper_x], upper_solution)
    raw_check = verify_grid_channel(epsilon, raw.grid, raw.outputs, raw.P_grid)
    progress("exact channel repair and average integration")
    channel = certify_continuous(raw, epsilon)
    if upper_x == lower_x and weights == pi:
        lower_solution = upper_solution
    else:
        progress(f"lower SOCP M={len(lower_x)}")
        lower_solution = solve_avg(epsilon, lower_x, pi, m2, progress=progress, **kwargs)
    progress("exact all-pattern lower-bound verification")
    # L_bin - D = sum(alpha) + mean_grid @ beta - m2(original prior).
    lower = certify_avg_dual(epsilon, lower_x, pi, m2, lower_solution["alpha"], lower_solution["beta"])
    U, L = avg_upper(prior, channel), F(lower["L_rational"])
    diagnostic_keys = ("primal_value", "dual_value", "primal_status", "dual_status",
                       "search_complete", "history", "pattern_mode", "runtime_seconds")
    result = seal(dict(kind="avg_result", epsilon=str(epsilon), prior=prior.as_dict(),
                       requested_m=m, partition=list(map(str, partition)),
                       within_bin_variance=str(D), channel=channel, lower=lower,
                       U_rational=str(U), U_cert=outward(U, True),
                       gap_rational=str(U-L), gap_cert=outward(U-L, True),
                       lower_bin_value_cert=outward(L+D, False),
                       upper_solver={k: upper_solution[k] for k in diagnostic_keys},
                       lower_solver={k: lower_solution[k] for k in diagnostic_keys},
                       raw_channel_check=raw_check, runtime_seconds=time.perf_counter()-started,
                       meaning="bounds on the given prior's average conditional variance, all x in [-1,1] unbiased and LDP"))
    verify_result(result)
    return result


def write_summary(out):
    rows = []
    for path in sorted(out.glob("*/result.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(case=path.parent.name, prior=d["prior"]["name"], model=d["prior"]["model"],
                         epsilon=d["epsilon"], requested_m=d["requested_m"],
                         M_upper=len(d["channel"]["grid"]), M_lower=len(d["lower"]["grid"]),
                         N_effective=d["channel"]["N_effective"], L_cert=d["lower"]["L_cert"],
                         U_cert=d["U_cert"], gap_cert=d["gap_cert"],
                         relative_gap=d["gap_cert"]/max(1e-30, abs(d["U_cert"])),
                         within_bin_variance=float(F(d["within_bin_variance"])),
                         upper_search_complete=d["upper_solver"]["search_complete"],
                         lower_search_complete=d["lower_solver"]["search_complete"],
                         runtime_seconds=d["runtime_seconds"]))
    if rows:
        with (out/"metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--prior", choices=["uniform", "histogram"], default="uniform")
    run.add_argument("--histogram", type=Path)
    run.add_argument("--hist-model", choices=["continuous", "discrete"], default="continuous")
    run.add_argument("--eps", nargs="+", type=float, default=[.1, .2, .5, 1, 2, 3])
    run.add_argument("--m", nargs="+", type=int, default=[5, 9, 17, 33])
    run.add_argument("--patterns", choices=["auto", "full", "cuts"], default="auto")
    run.add_argument("--max-cuts", type=int, default=100)
    run.add_argument("--solver-seconds", type=float, default=120.)
    run.add_argument("--gap-tol", type=float, default=0., help="stop refining each epsilon at this certified absolute gap (0 disables)")
    run.add_argument("--out", type=Path, default=Path("results/avg_uniform"))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--accelerated", action="store_true", help="fixed-prior dual checkpoints and robust weighted primal")
    run.add_argument("--no-symmetry", action="store_true", help="disable automatic exact prior/grid reflection test")
    run.add_argument("--seed-result", type=Path, help="verified same-prior result; reuse patterns and feasible channel only")
    verify = sub.add_parser("verify")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*/result.json"))
        if not paths:
            parser.error("no result.json files found")
        for path in paths:
            verify_result(json.loads(path.read_text(encoding="utf-8")))
            print(f"VERIFIED {path}", flush=True)
        return
    if (any(not np.isfinite(e) or not 0 < e <= 100 for e in args.eps)
            or min(args.m) < 2 or args.max_cuts < 1 or not np.isfinite(args.solver_seconds)
            or args.solver_seconds <= 0 or not np.isfinite(args.gap_tol) or args.gap_tol < 0):
        parser.error("invalid epsilon/grid/cut/time/gap settings")
    if (args.prior == "histogram") != (args.histogram is not None):
        parser.error("--prior histogram requires --histogram; uniform takes no histogram")
    if args.seed_result and not args.accelerated:
        parser.error("seed-result requires --accelerated")
    if args.prior == "uniform" and args.hist_model != "continuous":
        parser.error("uniform means continuous Uniform[-1,1]; use histogram for discrete priors")
    if args.histogram:
        data = json.loads(args.histogram.read_text(encoding="utf-8-sig"))
        prior = Prior(data["edges"], data["masses"], args.hist_model, data.get("name", args.histogram.stem))
    else:
        prior = Prior.uniform()
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "cvxpy", "clarabel")}
    save_json(args.out/"environment.json", versions)
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    failures = []
    sizes = sorted(set(args.m))
    if prior.model == "discrete":
        sizes = sizes[:1]
        print("Discrete histogram: grid uses all occupied bin centers; --m refinement is unnecessary.", flush=True)
    for eps in args.eps:
        for m in sizes:
            settings = dict(prior=prior.as_dict(), epsilon=eps, m=m, patterns=args.patterns,
                            max_cuts=args.max_cuts, solver_seconds=args.solver_seconds,
                            code_hash=code_hash, versions=versions)
            if args.accelerated:
                settings.update(accelerated=True, no_symmetry=args.no_symmetry,
                    seed_hash=hashlib.sha256(args.seed_result.read_bytes()).hexdigest() if args.seed_result else None)
            key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
            case = args.out/f"eps{eps:g}_m{m}_{key}"
            case.mkdir(exist_ok=True)
            result_path = case/"result.json"
            progress = lambda message: print(f"[eps={eps:g} m={m}] {message}", flush=True)
            try:
                if args.resume and result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    verify_result(result)
                    progress("resume: verified saved result")
                else:
                    save_json(case/"settings.json", settings)
                    if args.accelerated:
                        from .avg_refinement import run_case_accelerated
                        result = run_case_accelerated(prior, eps, m, args, progress, case)
                    else:
                        result = run_case(prior, eps, m, args, progress)
                    save_json(result_path, result)
                channel_from_certificate(result["channel"]).save(case/"channel.npz")
                (case/"error.json").unlink(missing_ok=True)
                progress(f"L={result['lower']['L_cert']:.12g} U={result['U_cert']:.12g} gap={result['gap_cert']:.4g} N={result['channel']['N_effective']}")
                write_summary(args.out)
                if args.gap_tol and result["gap_cert"] <= args.gap_tol:
                    progress("certified gap target reached")
                    break
            except Exception as exc:
                progress(f"FAILED: {type(exc).__name__}: {exc}")
                save_json(case/"error.json", dict(error_type=type(exc).__name__, message=str(exc)))
                failures.append(str(case))
    write_summary(args.out)
    if failures:
        raise SystemExit(f"{len(failures)} cases failed; see error.json")


if __name__ == "__main__":
    main()
