"""Verify saved lower bounds or construct new channels with the paper solvers."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
if hasattr(sys,"set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from reproduce import DATA, DATASETS, read_csv, read_json
import numpy as np


def verify_lower(epsilon, objective):
    from src.avg_prior import Prior, uniform_grid
    from src.avg_solver import verify_avg_dual
    from src.certify_dual import verify_dual

    worst = read_csv(DATA/"worst_bounds.csv") if objective != "avg" else []
    for row in worst:
        e = float(row["epsilon"])
        if epsilon is not None and e != epsilon:
            continue
        cert = read_json(DATA/"certificates/worst"/f"eps{e:g}.json")
        verify_dual(cert)
        if float(row["L"]) != cert["L_cert"] or e != float(cert["epsilon"]):
            raise ValueError("Worst-case certificate differs from the bound used in the table.")
        print(f"Verified CMV-W lower bound at eps={e:g}",flush=True)
    average = read_csv(DATA/"avg_bounds.csv") if objective != "worst" else []
    for row in average:
        ds,e = row["dataset"],float(row["epsilon"])
        if epsilon is not None and e != epsilon:
            continue
        context = read_json(DATA/"certificates/avg"/f"{ds}_eps{e:g}.json")
        cert = context["certificate"]
        prior = Prior(**read_json(DATA/"priors"/f"{ds}.json"))
        x,w,d = prior.coarsen(uniform_grid(int(row["requested_M"])))
        if (list(map(str,x)) != cert["grid"] or list(map(str,w)) != cert["weights"]
                or d != Fraction(context["within_bin_variance"])
                or prior.moments()[2] != Fraction(cert["second_moment"])
                or e != float(cert["epsilon"]) or float(row["L"]) != cert["L_cert"]):
            raise ValueError("Wrong prior or bound in AVG certificate.")
        verify_avg_dual(cert)
        print(f"Verified CMV-D lower bound for {ds} at eps={e:g}",flush=True)


def solve_avg(args):
    from src.avg_prior import Prior
    from src.avg_refinement import run_case_accelerated
    from src.certify_continuous import channel_from_certificate
    options = SimpleNamespace(patterns="cuts",max_cuts=300,solver_seconds=args.seconds,
                              no_symmetry=False,seed_result=None)
    prior = Prior.uniform() if args.prior == "uniform" else Prior(**read_json(DATA/"priors"/f"{args.prior}.json"))
    args.out.mkdir(parents=True,exist_ok=True)
    result = run_case_accelerated(prior,args.eps,args.m,options,
                                  lambda value:print(value,flush=True),args.out)
    (args.out/"result.json").write_text(json.dumps(result),encoding="utf-8")
    channel_from_certificate(result["channel"]).save(args.out/"channel.npz")
    print(f"L={result['lower']['L_cert']:.9g}, U={result['U_cert']:.9g}")


def solve_aaa(args):
    from mechanisms.a3m import audit_mechanism, conditional_noise_moments, optimize_a3m
    from src.avg_prior import Prior, uniform_grid
    prior = Prior(**read_json(DATA/"priors"/f"{args.prior}.json"))
    grid = uniform_grid(17)
    mass = np.asarray(prior.hat_weights(grid),float)
    args.out.mkdir(parents=True,exist_ok=True)
    for ratio in (4,8,16,32,64):
        try:
            channel = optimize_a3m(mass,args.eps,noise_range_ratio=ratio,tail_ratio=.5,
                solver_options=dict(time_limit=args.seconds,primal_feasibility_tolerance=1e-9,
                                    dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-10))
        except RuntimeError:
            continue
        diag = audit_mechanism(channel)
        if (diag["max_normalization_error"] > 1e-8 or diag["max_noise_mean_error"] > 1e-8
                or diag["min_probability"] < -1e-10 or diag["max_ldp_violation"] > 1e-9):
            continue
        x = np.asarray(channel["input_points"])
        mu,second = conditional_noise_moments(channel)
        np.savez_compressed(args.out/"channel.npz",grid=x,mean=x+mu,second=x*x+2*x*mu+second,
                            core_probabilities=channel["core_probabilities"],tail_ratio=channel["tail_ratio"],
                            noise_half_width=channel["noise_half_width"],sigma=channel["sigma"])
        print(f"AAA range ratio={ratio:g}")
        return
    raise RuntimeError("No audited AAA candidate for range ratios 4 through 64.")


def solve_n_output(args):
    from mechanisms import optimize_worst_exp as optimizer
    from mechanisms.noutput_safety import audit_arrays, repair_arrays, worst_squared_error

    args.out.mkdir(parents=True,exist_ok=True)
    candidates = []
    for start in range(args.starts):
        np.random.seed(start)
        strategy = "pm_style" if start == 0 else "pm_noise"
        result = optimizer.optimize(args.eps-1e-6,args.n,ftol=1e-10,maxiter=args.maxiter,
                                    initial_guess_strategy=strategy)
        half = args.n//2
        model = (optimizer.WorstCaseOptimizerWithA0_FreeA if args.n%2
                 else optimizer.WorstCaseOptimizerNoA0_FreeA)(half,args.eps-1e-6)
        a,p = model.unpack_vars(result.x)
        a,p,_ = repair_arrays(a,p,args.eps,prune_threshold=1e-10,privacy_margin=1e-6)
        if audit_arrays(a,p,args.eps).ldp_safe:
            candidates.append((worst_squared_error(a,p),a,p))
    if not candidates:
        raise RuntimeError("No feasible candidate.")
    value,a,p = min(candidates,key=lambda item:item[0])
    grid = a@p
    grid[0],grid[-1] = 0.,1.
    np.savez_compressed(args.out/"channel.npz",grid=np.r_[-grid[:0:-1],grid],
                        outputs=a,P_grid=np.c_[p[::-1,:0:-1],p].T)
    print(f"Worst variance={value:.9g}, N={len(a)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    verifier = sub.add_parser("verify-lower")
    verifier.add_argument("--eps",type=float)
    verifier.add_argument("--objective",choices=["both","worst","avg"],default="both")
    avg = sub.add_parser("avg")
    avg.add_argument("--prior",choices=["uniform",*DATASETS],required=True)
    avg.add_argument("--eps",type=float,required=True)
    avg.add_argument("--m",type=int,default=65)
    avg.add_argument("--seconds",type=float,default=600)
    avg.add_argument("--out",type=Path,required=True)
    aaa = sub.add_parser("aaa")
    aaa.add_argument("--prior",choices=DATASETS,required=True)
    aaa.add_argument("--eps",type=float,required=True)
    aaa.add_argument("--seconds",type=float,default=120)
    aaa.add_argument("--out",type=Path,required=True)
    discrete = sub.add_parser("n-output")
    discrete.add_argument("--n",type=int,required=True)
    discrete.add_argument("--eps",type=float,required=True)
    discrete.add_argument("--starts",type=int,default=5)
    discrete.add_argument("--maxiter",type=int,default=3000)
    discrete.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.action == "verify-lower":
        verify_lower(args.eps,args.objective)
    elif args.action == "avg":
        solve_avg(args)
    elif args.action == "aaa":
        solve_aaa(args)
    else:
        solve_n_output(args)


if __name__ == "__main__":
    main()
