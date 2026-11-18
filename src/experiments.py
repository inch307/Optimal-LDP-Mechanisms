import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction as F

import numpy as np
import psutil
import cvxpy as cp
from .grid import make_grid, refine_grid
from .dual_b import solve_B_dual
from .primal_b import solve_B_primal
from .certify_dual import certify_dual, verify_dual
from .channel import GridChannel, from_moments, merge_duplicate_outputs, verify_grid_channel
from .recover_contact import recover_from_dual
from .extend_piecewise import continuous_worst, interpolate_channel
from .certify_continuous import certify_continuous, channel_from_certificate, verify_continuous
from .exact import save_json, outward


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class MemoryMonitor:
    def __enter__(self):
        self.peak = 0
        self.stop = threading.Event()
        def sample():
            while not self.stop.is_set():
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
                self.stop.wait(.1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def run_experiment(config):
    folder = Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    eps = config["epsilon"]
    initial = make_grid(config["initial_m"], config["grid_type"])
    grid = initial
    rows, attempts = [], []
    start, cpu_start = time.perf_counter(), time.process_time()
    rounds = config["rounds"] if config["mode"] in ("all", "adaptive") else 1
    stop_reason = "max_rounds"
    last_summary = {}
    def status(message):
        save_json(folder/"progress.json", dict(stage=message, elapsed=time.perf_counter()-start))
    with MemoryMonitor() as memory:
        for iteration in range(rounds):
            if len(grid) > config["max_m"]:
                break
            prefix = f"r{iteration:02d}_m{len(grid)}"
            pattern_mode = config["patterns"]
            if pattern_mode == "auto":
                pattern_mode = "full" if len(grid) <= config["full_max_m"] else "cuts"
            status(f"round {iteration+1}/{rounds}, M={len(grid)}, dual/{pattern_mode}")
            accelerated = config.get("accelerated", False)
            symmetry = accelerated and not config.get("no_symmetry", False)
            if accelerated and config.get("reuse_dual_only"):
                from .acceleration import lift_seed
                tick = time.perf_counter()
                seed = lift_seed(config["seed_dual"],eps,grid)
                cert = certify_dual(eps,grid,seed["alpha"],seed["beta"],seed["gamma"],method="interval")
                dual = {k:seed[k] for k in ("alpha","beta","gamma","bits")}
                dual.update(certificate=cert,L_float=cert["L_cert"],L_cert=cert["L_cert"],
                            solver_status="certified_seed",search_complete=False,pattern_mode="cuts",
                            runtime=time.perf_counter()-tick,history=[],stop_reason="reuse_dual_only",
                            seed={k:seed[k] for k in ("source","source_M","source_L")})
            elif accelerated:
                from .dual_accelerated import solve_B_dual_accelerated
                def checkpoint(candidate):
                    # Save before further search/primal work, including hard-timeout cases.
                    save_json(folder/(prefix+"_dual_cert.json"), candidate["certificate"])
                    target = folder/(prefix+"_dual_float.npz")
                    temporary = target.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, grid=grid,
                                        **{k:candidate[k] for k in ("alpha","beta","gamma","bits")})
                    temporary.replace(target)
                dual = solve_B_dual_accelerated(
                    eps, grid, time_limit=config["solver_seconds"], max_cuts=config["max_cuts"],
                    seed_path=config.get("seed_dual"), symmetry=symmetry,
                    pattern_limit=None if config.get("pattern_limit", -1) == -1 else config["pattern_limit"],
                    staged=not config.get("no_staged_precision", False), checkpoint=checkpoint,
                    progress=lambda s: status(f"round {iteration+1}/{rounds}, M={len(grid)}, {s}"))
            else:
                dual = solve_B_dual(eps, grid, pattern_mode, time_limit=config["solver_seconds"],
                                    max_cuts=config["max_cuts"], progress=lambda s: status(f"round {iteration+1}/{rounds}, M={len(grid)}, {s}"))
            np.savez_compressed(folder/(prefix+"_dual_float.npz"), grid=grid,
                                **{k:dual[k] for k in ("alpha","beta","gamma","bits")})
            save_json(folder/(prefix+"_dual_search.json"), {k:v for k,v in dual.items() if k not in ("alpha","beta","gamma","bits","certificate")})
            status(f"round {iteration+1}/{rounds}, M={len(grid)}, exact dual certificate")
            cert_start = time.perf_counter()
            if accelerated:
                lower = dual["certificate"]
                verify_dual(lower)
                save_json(folder/(prefix+"_dual_cert.json"), lower)
            else:
                lower = certify_dual(eps, grid, dual["alpha"], dual["beta"], dual["gamma"],
                                     method=config["certificate_method"], path=folder/(prefix+"_dual_cert.json"))
            lower_seconds = time.perf_counter()-cert_start
            candidates = []
            precertified = {}
            if accelerated and config.get("seed_channel"):
                with np.load(config["seed_channel"], allow_pickle=False) as saved:
                    old_channel = interpolate_channel(saved["grid"], saved["outputs"], saved["P_grid"])
                    reused = GridChannel(grid, old_channel.outputs, old_channel.probabilities(grid))
                reused = merge_duplicate_outputs(reused)
                candidates.append(("reused", reused, "reused_channel_recertified", None, 0.))
                # Preserve a usable upper before spending hours on a new primal.
                upper_start = time.perf_counter()
                precertified["reused"] = (certify_continuous(reused, eps,
                    path=folder/(prefix+"_reused_continuous_cert.json"), symmetry=symmetry),
                    time.perf_counter()-upper_start)
            # Always retain the unrestricted-moment primal comparison (restricted
            # columns in cuts mode). Neither a restricted objective nor a raw
            # dual solver value is reported as a certified lower bound.
            status(f"round {iteration+1}/{rounds}, M={len(grid)}, moment primal")
            try:
                if config.get("robust_primal"):
                    from .primal_robust import solve_B_primal_robust
                    def primal_checkpoint(candidate):
                        save_json(folder/(prefix+"_moment_continuous_cert.json"),candidate["certificate"])
                        target = folder/(prefix+"_primal_float.npz")
                        temporary = target.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary,grid=grid,
                                            **{k:candidate[k] for k in ("w","z","r","bits")})
                        temporary.replace(target)
                        channel_from_certificate(candidate["certificate"]).save(folder/(prefix+"_moment_certified_channel_float.npz"))
                    primal = solve_B_primal_robust(eps,grid,dual["bits"],time_limit=config["solver_seconds"],
                        symmetry=symmetry,lower_bound=lower["L_rational"],
                        log_path=folder/(prefix+"_primal_attempts.json"), checkpoint=primal_checkpoint,
                        progress=lambda s: status(f"round {iteration+1}/{rounds}, M={len(grid)}, {s}"))
                    precertified["moment"] = (primal["certificate"],
                        sum(r.get("certificate_seconds",0.) for r in primal["attempts"]["attempts"]))
                else:
                    primal = solve_B_primal(eps, grid, pattern_mode, bits=dual["bits"], time_limit=config["solver_seconds"], symmetry=symmetry)
                np.savez_compressed(folder/(prefix+"_primal_float.npz"), grid=grid,
                                    **{k:primal[k] for k in ("w","z","r","bits")})
                candidates.append(("moment", from_moments(eps,grid,primal), primal["solver_status"], primal["tau"], primal["runtime"]))
                if accelerated and "moment" not in precertified:
                    # A long contact-LP attempt must not lose an already feasible upper.
                    upper_start = time.perf_counter()
                    precertified["moment"] = (certify_continuous(merge_duplicate_outputs(candidates[-1][1]), eps,
                        path=folder/(prefix+"_moment_continuous_cert.json"), symmetry=symmetry),
                        time.perf_counter()-upper_start)
            except (ValueError, RuntimeError, cp.error.SolverError) as exc:
                if not accelerated:
                    raise
                attempts.append(dict(round=iteration, M=len(grid), method="moment", status=str(exc)))
            status(f"round {iteration+1}/{rounds}, M={len(grid)}, contact LP")
            try:
                if config.get("skip_contact"):
                    recovered = dict(channel=None,status="skipped",tau=None,runtime=0.,attempts=[])
                else:
                    recovered = recover_from_dual(eps,grid,dual["alpha"],dual["beta"],dual["gamma"],dual["bits"],
                                                  time_limit=config["solver_seconds"])
            except (ValueError, RuntimeError) as exc:
                if not accelerated:
                    raise
                recovered = dict(channel=None, status="failed", tau=None, runtime=0.,
                                 attempts=[dict(error=str(exc))])
            save_json(folder/(prefix+"_contact_attempts.json"), {k:v for k,v in recovered.items() if k != "channel"})
            if recovered["channel"] is not None:
                candidates.append(("contact",recovered["channel"],recovered["status"],recovered["tau"],recovered["runtime"]))
            elif recovered["status"] != "skipped":
                attempts.append(dict(round=iteration, M=len(grid), method="contact", status="LP_failed"))
                rows.append(dict(epsilon=eps,initial_M=len(initial),grid_type=config["grid_type"],round=iteration,
                                 M=len(grid),method="contact",mode="recovery_failed",pattern_mode=pattern_mode,
                                 patterns=len(dual["bits"]),L_float=dual["L_float"],L_cert=lower["L_cert"],
                                 primal_status="LP_failed",dual_status=dual["solver_status"],dual_verify="passed",
                                 channel_verify="not_available",optimality_status="no channel from contact LP",
                                 cumulative_wall_seconds=time.perf_counter()-start))
            certified_candidates = []
            for method,raw,solver_status,objective,solve_seconds in candidates:
                status(f"round {iteration+1}/{rounds}, M={len(grid)}, {method}: exact continuous certificate")
                raw = merge_duplicate_outputs(raw)
                raw.save(folder/(prefix+f"_{method}_raw_channel.npz"))
                residuals = verify_grid_channel(eps,grid,raw.outputs,raw.P_grid)
                upper_start = time.perf_counter()
                try:
                    if method in precertified:
                        upper = precertified[method][0]
                    else:
                        upper = certify_continuous(raw,eps,path=folder/(prefix+f"_{method}_continuous_cert.json"), symmetry=symmetry)
                except ValueError as exc:
                    attempts.append(dict(round=iteration,M=len(grid),method=method,status="upper_certificate_failed",error=str(exc)))
                    rows.append(dict(epsilon=eps,initial_M=len(initial),grid_type=config["grid_type"],round=iteration,
                                     M=len(grid),method=method,mode="upper_certificate_failed",L_cert=lower["L_cert"],
                                     primal_status=solver_status,dual_verify="passed",channel_verify="failed",
                                     optimality_status="no certified upper",error=str(exc)))
                    continue
                upper_seconds = precertified[method][1] if method in precertified else time.perf_counter()-upper_start
                channel = channel_from_certificate(upper)
                channel.save(folder/(prefix+f"_{method}_certified_channel_float.npz"))
                # Float export is for deployment/diagnostics. Exact JSON is authoritative.
                evaluation = continuous_worst(channel)
                samples = np.linspace(-1,1,config["samples"])
                sample_max = float(max(interpolate_channel(channel.grid,channel.outputs,channel.P_grid).variance(samples)))
                lower_q, ug, uc = F(lower["L_rational"]),F(upper["U_grid_rational"]),F(upper["U_cont_rational"])
                if lower_q > ug:
                    raise RuntimeError("independent exact lower bound exceeds feasible grid upper")
                base = dict(epsilon=eps,initial_M=len(initial),grid_type=config["grid_type"],round=iteration,
                            M=len(grid),Delta=evaluation["Delta"],method=method,pattern_mode=pattern_mode,
                            patterns=len(dual["bits"]),N_effective=upper["N_effective"],N_raw=len(raw.outputs),
                            L_float=dual["L_float"],L_cert=lower["L_cert"],primal_objective_float=objective,
                            U_grid_raw=residuals["U_grid"],U_grid=upper["U_grid_cert"],
                            gap_grid=outward(ug-lower_q,True),gap_grid_scaled=outward((ug-lower_q)/max(F(1),abs(lower_q)),True),
                            primal_status=solver_status,dual_status=dual["solver_status"],
                            dual_search_complete=dual["search_complete"],dual_verify="passed",channel_verify="passed",
                            normalization_residual=residuals["normalization"],unbiasedness_residual=residuals["unbiasedness"],
                            ldp_residual=residuals["ldp"],negative_residual=residuals["negativity"],
                            privacy_repair_mass=float(F(upper["privacy_mix"])),mean_repair_mass=float(F(upper["mean_repair_mass"])),
                            dual_seconds=dual["runtime"],primal_seconds=solve_seconds,lower_cert_seconds=lower_seconds,
                            upper_cert_seconds=upper_seconds,cumulative_wall_seconds=time.perf_counter()-start,
                            cumulative_cpu_seconds=time.process_time()-cpu_start,peak_memory_mb=memory.peak/1024**2,
                            lower_certificate=prefix+"_dual_cert.json",upper_certificate=prefix+f"_{method}_continuous_cert.json")
                continuous = dict(U_cont=upper["U_cont_cert"],gap_cont=outward(uc-lower_q,True),
                                  gap_cont_scaled=outward((uc-lower_q)/max(F(1),abs(lower_q)),True),
                                  U_sample=sample_max,sampling_miss=max(0.,upper["U_cont_cert"]-sample_max))
                modes = (["A","B","C"] if iteration == 0 else ["C"]) if config["mode"] == "all" else [{"grid":"A","interpolated":"B","adaptive":"C"}[config["mode"]]]
                for mode in modes:
                    row = {**base,"mode":mode}
                    if mode != "A":
                        row.update(continuous)
                    gap = ug-lower_q if mode == "A" else uc-lower_q
                    near = gap <= F(str(config["gap_abs"]))+F(str(config["gap_rel"]))*max(F(1),abs(lower_q))
                    row["optimality_status"] = ("B-global on G (within tolerance)" if mode == "A" else "near-global for continuous (within tolerance)") if near else "certified bounds; target gap not reached"
                    rows.append(row)
                certified_candidates.append((uc,channel,evaluation,method))
            write_csv(folder/"metrics.csv",rows)
            save_json(folder/"failures.json",attempts)
            if not certified_candidates:
                raise RuntimeError("no candidate passed exact upper verification")
            best = min(certified_candidates,key=lambda r:r[0])
            gap = best[0]-F(lower["L_rational"])
            threshold = F(str(config["gap_abs"]))+F(str(config["gap_rel"]))*max(F(1),abs(F(lower["L_rational"])))
            last_summary = dict(final_M=len(grid),rounds_completed=iteration+1,best_method=best[3],
                                L_cert=lower["L_cert"],U_cont=outward(best[0],True),gap_cont=outward(gap,True),
                                gap_threshold=float(threshold),target_reached=gap<=threshold,
                                dual_verify="passed",channel_verify="passed")
            if config.get("robust_primal"):
                moments = [r for r in certified_candidates if r[3] == "moment"]
                previous = precertified.get("reused")
                last_summary["primal_repair"] = dict(
                    status="new_primal_certified" if moments else "previous_channel_only",
                    new_U_cont=outward(moments[0][0],True) if moments else None,
                    reused_U_cont=previous[0]["U_cont_cert"] if previous else None,
                    improved_over_reused=bool(moments and previous and moments[0][0] < F(previous[0]["U_cont_rational"])),
                    selected_solver=primal.get("solver") if moments else None)
            if gap <= threshold and not config.get("no_early_stop",False):
                stop_reason = "target_gap"
                status(f"target certified gap reached after round {iteration+1}")
                break
            room = config["max_m"]-len(grid)
            if room <= 0:
                stop_reason = "max_M"
                break
            if iteration+1 == rounds:
                stop_reason = "max_rounds" if config["mode"] in ("all","adaptive") else "single_grid"
                break
            new = refine_grid(grid,best[2]["maximizers"],min(config["add_points"],room))
            if len(new) == len(grid):
                stop_reason = "grid_stagnation"
                break
            grid = new
    save_json(folder/"done.json",dict(status="completed",wall_seconds=time.perf_counter()-start,
                                      cpu_seconds=time.process_time()-cpu_start,peak_memory_mb=memory.peak/1024**2,
                                      rows=len(rows),config=config,stop_reason=stop_reason,**last_summary))
    status(f"completed: M={last_summary.get('final_M')}, stop={stop_reason}")


def collect(root):
    rows = []
    for path in sorted(Path(root).glob("**/cases/*/metrics.csv")):
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["case"] = str(path.parent.relative_to(root))
                rows.append(row)
    write_csv(Path(root)/"metrics.csv",rows)
    # Direct full/cuts and M/runtime comparisons remain visible in this complete table.
    if rows:
        lines = ["# Experiment results", "", "All probabilities use P_grid shape (M,K). A has no continuous upper claim.", "",
                 "|case|mode|round|M|method|L certified|U grid|U continuous|continuous gap|seconds|",
                 "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append("|"+"|".join(str(r.get(k,"")) for k in ("case","mode","round","M","method","L_cert","U_grid","U_cont","gap_cont","cumulative_wall_seconds"))+"|")
        (Path(root)/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return len(rows)


def verify_directory(root):
    files = sorted(Path(root).rglob("*_cert.json"))
    if not files:
        raise ValueError("no certificate files found")
    failed = []
    last_log = time.perf_counter()
    for index,path in enumerate(files):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            (verify_dual if data["kind"] == "dual" else verify_continuous)(data)
        except Exception as exc:
            failed.append(dict(path=str(path),error=str(exc)))
        if time.perf_counter()-last_log >= 30:
            print(f"Verify {index+1}/{len(files)} ({100*(index+1)/len(files):.0f}%), failed={len(failed)}",flush=True)
            last_log=time.perf_counter()
    save_json(Path(root)/"independent_verification.json",dict(checked=len(files),passed=len(files)-len(failed),failures=failed))
    print(f"Independent verification: {len(files)-len(failed)}/{len(files)} passed",flush=True)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Unrestricted-output grid moment LDP, contact recovery and exact certificates")
    subs = parser.add_subparsers(dest="command",required=True)
    run = subs.add_parser("run")
    run.add_argument("--eps",type=float,nargs="+",default=[.1,.2,.5,1.,1.5,2.,2.5,3.])
    run.add_argument("--m",type=int,nargs="+",default=[3,5,7])
    run.add_argument("--grid",choices=["uniform","nonuniform"],nargs="+",default=["uniform"])
    run.add_argument("--mode",choices=["all","grid","interpolated","adaptive"],default="all")
    run.add_argument("--patterns",choices=["auto","full","cuts"],default="auto")
    run.add_argument("--full-max-m",type=int,default=9)
    run.add_argument("--max-m",type=int,default=33)
    run.add_argument("--max-cuts",type=int,default=60)
    run.add_argument("--rounds",type=int,default=4)
    run.add_argument("--add-points",type=int,default=3)
    run.add_argument("--certificate-method",choices=["auto","exhaustive","interval"],default="auto")
    run.add_argument("--solver-seconds",type=float,default=120.)
    run.add_argument("--accelerated",action="store_true",help="certified incumbents, symmetry, working set and staged precision")
    run.add_argument("--robust-primal",action="store_true",help="certified primal retries with native solver diagnostics")
    run.add_argument("--reuse-dual-only",action="store_true",help="verify/import seed dual without another dual search")
    run.add_argument("--skip-contact",action="store_true",help="omit the optional contact LP recovery")
    run.add_argument("--seed-dual",help="verified certificate on an exact subset grid")
    run.add_argument("--seed-channel",help="previous certified float channel; always recertified")
    run.add_argument("--no-symmetry",action="store_true")
    run.add_argument("--no-staged-precision",action="store_true")
    run.add_argument("--pattern-limit",type=int,default=-1,help="soft cap; -1 auto, 0 disables deletion")
    run.add_argument("--case-seconds",type=float,default=1800.,help="hard subprocess timeout for one eps/M/grid case")
    run.add_argument("--log-interval",type=float,default=30.)
    run.add_argument("--gap-abs",type=float,default=1e-5)
    run.add_argument("--gap-rel",type=float,default=1e-5)
    run.add_argument("--no-early-stop",action="store_true",help="continue grid refinement after reaching target gap")
    run.add_argument("--samples",type=int,default=1001)
    run.add_argument("--out",type=Path,default=Path("results/b_dual_primal"))
    run.add_argument("--resume",action="store_true")
    run.add_argument("--shard-index",type=int,default=0)
    run.add_argument("--shard-count",type=int,default=1)
    for name in ("verify","summarize","compare"):
        sub = subs.add_parser(name)
        sub.add_argument("path",type=Path)
    worker = subs.add_parser("_case",help=argparse.SUPPRESS)
    worker.add_argument("config",type=Path)
    args = parser.parse_args()
    if args.command == "_case":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        try:
            run_experiment(config)
        except Exception as exc:
            save_json(Path(config["folder"])/"error.json",dict(error=str(exc),traceback=traceback.format_exc()))
            raise
        return
    if args.command == "verify":
        verify_directory(args.path)
        return
    if args.command in ("summarize","compare"):
        args.path.mkdir(parents=True,exist_ok=True)
        print(f"Collected {collect(args.path)} rows")
        print(f"Comparison: {(args.path/'comparison.html').resolve()}")
        return
    if (any(not 0 < e <= 100 for e in args.eps) or any(m<3 or m>args.max_m for m in args.m)
        or args.rounds<1 or args.add_points<1 or args.samples<2 or args.max_cuts<1
        or not 0<=args.shard_index<args.shard_count or args.case_seconds<=0 or args.solver_seconds<=0
        or args.log_interval<1 or args.gap_abs<0 or args.gap_rel<0):
        parser.error("invalid epsilon/M/budget/tolerance/shard settings")
    if args.pattern_limit < -1 or (args.accelerated and (args.patterns != "cuts" or args.rounds != 1)):
        parser.error("accelerated mode requires --patterns cuts --rounds 1; pattern-limit >= -1")
    if (args.robust_primal or args.reuse_dual_only) and not args.accelerated:
        parser.error("robust-primal/reuse-dual-only require --accelerated")
    if args.reuse_dual_only and not args.seed_dual:
        parser.error("reuse-dual-only requires --seed-dual")
    args.out.mkdir(parents=True,exist_ok=True)
    versions = {name:importlib.metadata.version(name) for name in ("numpy","scipy","cvxpy","clarabel","scs","psutil")}
    save_json(args.out/f"environment_shard{args.shard_index}.json",dict(python=sys.version,platform=platform.platform(),versions=versions,
                                                                        threads=os.environ.get("NOUTPUT_THREADS","1"),argv=sys.argv))
    jobs = [(e,m,g) for e in args.eps for m in args.m for g in args.grid]
    jobs = [job for i,job in enumerate(jobs) if i%args.shard_count==args.shard_index]
    start, failures = time.perf_counter(), []
    entry = Path(__file__).resolve().parents[1]/"run_worst.py"
    source_hasher = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hasher.update(source.name.encode())
        source_hasher.update(source.read_bytes())
    print(f"START {len(jobs)} cases | shard {args.shard_index+1}/{args.shard_count} | progress every {args.log_interval:g}s",flush=True)
    for index,(eps,m,kind) in enumerate(jobs):
        config = {k:v for k,v in vars(args).items() if k not in ("eps","m","grid","out","command","resume","shard_index","shard_count","log_interval","case_seconds")}
        config.update(epsilon=eps,initial_m=m,grid_type=kind,source_sha256=source_hasher.hexdigest())
        digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:10]
        folder = args.out/"cases"/f"eps{eps:g}_m{m}_{kind}_{digest}"
        config["folder"] = str(folder.resolve())
        if (folder/"done.json").exists():
            if args.resume:
                print(f"[{index+1}/{len(jobs)}] cached eps={eps:g} M={m} {kind}",flush=True)
                continue
            parser.error(f"completed case exists: {folder}; use --resume or another --out")
        folder.mkdir(parents=True,exist_ok=True)
        for stale in ("error.json","timeout.json","metrics.csv","progress.json","failures.json"):
            (folder/stale).unlink(missing_ok=True)
        save_json(folder/"config.json",config)
        case_start = last_log = time.perf_counter()
        print(f"[{index+1}/{len(jobs)}] start eps={eps:g} M={m} {kind}",flush=True)
        with (folder/"worker.log").open("w",encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable,str(entry),"_case",str(folder/"config.json")],stdout=log,stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            timed_out = False
            try:
                while proc.poll() is None:
                    elapsed = time.perf_counter()-case_start
                    if elapsed > args.case_seconds:
                        proc.kill()
                        proc.wait()
                        timed_out = True
                        save_json(folder/"timeout.json",dict(hard_limit_seconds=args.case_seconds,elapsed=elapsed))
                        break
                    if time.perf_counter()-last_log >= args.log_interval:
                        stage = "starting"
                        try:
                            stage = json.loads((folder/"progress.json").read_text(encoding="utf-8"))["stage"]
                        except (OSError,ValueError):
                            pass
                        print(f"[{index+1}/{len(jobs)}; completed {100*index/max(1,len(jobs)):.0f}%] eps={eps:g} | {stage} | case {elapsed:.0f}s / total {time.perf_counter()-start:.0f}s",flush=True)
                        last_log = time.perf_counter()
                    try:
                        proc.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                collect(args.out)
                raise
        ok = proc.returncode==0 and not timed_out
        label = "done" if ok else "TIMEOUT" if timed_out else "FAILED"
        detail = f"initial_M={m}"
        if ok:
            done = json.loads((folder/"done.json").read_text(encoding="utf-8"))
            detail += (f" -> M={done['final_M']} rounds={done['rounds_completed']}"
                       f" | cert=passed L={done['L_cert']:.9g} U={done['U_cont']:.9g}"
                       f" gap={done['gap_cont']:.3g} stop={done['stop_reason']}")
            if "primal_repair" in done:
                detail += f" primal={done['primal_repair']['status']}"
        print(f"[{index+1}/{len(jobs)}; {100*(index+1)/max(1,len(jobs)):.0f}%] {label} eps={eps:g} {detail} | {time.perf_counter()-case_start:.1f}s | total {time.perf_counter()-start:.1f}s",flush=True)
        if not ok:
            failures.append(str(folder))
        collect(args.out)
    count = collect(args.out)
    save_json(args.out/f"run_status_shard{args.shard_index}.json",dict(failed_cases=failures,rows=count,wall_seconds=time.perf_counter()-start))
    print(f"FINISH rows={count}, failed={len(failures)}, elapsed={time.perf_counter()-start:.1f}s; {args.out.resolve()}",flush=True)
    if failures:
        raise SystemExit(1)
