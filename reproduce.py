"""Recompute the evaluation tables and Figure 2 from the paper's saved designs."""
import argparse
import csv
from fractions import Fraction
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np

from baselines import categorical_moments, continuous
from stress import Moments, expected as synthetic_mse, integrate, neighborhood


DATA = Path(__file__).resolve().parent / "data"
EPS = (.1, .2, .5, 1., 1.5, 2., 2.5, 3.)
DATASETS = {"adult":"Adult", "credit":"Credit", "bh_price":"Beijing housing", "ONP_wc":"Online News Popularity"}
BASELINES = ("SR", "TO", "PM", "PM-opt", "OGPM-U", "SW")
NUMERICAL = BASELINES[2:]
CATEGORICAL = ("GRR", "OUE")
GENERAL = BASELINES + CATEGORICAL + ("CMV-U", "CMV-W")
CAPS = NUMERICAL + CATEGORICAL + ("N-output",)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def channel(grid, outputs, p):
    return Moments(dict(grid=grid,mean=p@outputs,second=p@(outputs**2),kind="linear"))


def saved_n_output(e, bits):
    with np.load(DATA/"n_output"/f"eps{e:g}_bits{bits}.npz",allow_pickle=False) as z:
        return channel(z["grid"],z["outputs"],z["P_grid"])


def finite_baselines(e):
    from mechanisms.topm import TOPM
    grid = np.array([-1.,0.,1.])
    a = 1/np.tanh(e/2)
    plus = (1+grid/a)/2
    sr = channel(grid,np.array([-a,a]),np.c_[1-plus,plus])
    to = TOPM(1,e,np.random.default_rng(0),k=1)
    zero = to.P0_0+to.Px_0*abs(grid)
    plus = (1-zero+grid/to.C)/2
    return {"SR":sr, "TO":channel(grid,np.array([-to.C,0.,to.C]),np.c_[1-zero-plus,zero,plus])}


def polynomial_moments(mechanism):
    x = np.array([-1.,0.,1.])
    _, second = mechanism.moments(x)
    a,b,c = second
    return Moments(dict(kind="quadratic",coefficients=[b,(c-a)/2,(a+c)/2-b]))


def load_records():
    records = {}
    for item in read_json(DATA/"datasets.json"):
        with np.load(DATA/"datasets"/f"{item['dataset']}.npz",allow_pickle=False) as z:
            x,counts,offsets = z["x"],z["counts"],z["offsets"]
        n,d = item["n"],item["d"]
        np.testing.assert_array_equal(np.add.reduceat(counts,offsets[:-1]),np.full(d,n))
        records[item["dataset"]] = (x,counts,offsets,n)
    return records


def expected_mean_mse(fn, records):
    x,counts,offsets,n = records
    mu,second = fn(x)
    variance = np.maximum(second-mu**2,0)
    bias = np.add.reduceat(counts*(mu-x),offsets[:-1])/n
    return float(np.mean(np.add.reduceat(counts*variance,offsets[:-1])/n**2+bias**2))


def prior_components(ds):
    prior = read_json(DATA/"priors"/f"{ds}.json")
    edges = [float(Fraction(x)) for x in prior["edges"]]
    masses = [float(Fraction(x)) for x in prior["masses"]]
    return [(a,b,p) for a,b,p in zip(edges[:-1],edges[1:],masses) if p]


def reduction(ours, baseline):
    return 100*(1-ours/baseline)


def calculate():
    records = load_records()
    prior = {ds:prior_components(ds) for ds in DATASETS}
    saved = read_json(DATA/"moments.json")
    bounds = {float(r["epsilon"]):(float(r["L"]),float(r["U"])) for r in read_csv(DATA/"worst_bounds.csv")}
    avg_bounds = {(r["dataset"],float(r["epsilon"])):(float(r["L"]),float(r["U"]))
                  for r in read_csv(DATA/"avg_bounds.csv")}
    results = dict(worst={}, stress={}, prior={}, mse={})
    for e in EPS:
        spec = saved[f"{e:g}"]
        continuous_mechanisms = {name:continuous(name,e) for name in NUMERICAL}
        common = finite_baselines(e) | {n:polynomial_moments(m) for n,m in continuous_mechanisms.items()}
        common |= {"CMV-W":Moments(spec["Optimal mech-W"]), "CMV-U":Moments(spec["Optimal mech-U"])}
        legacy = {b:saved_n_output(e,b) for b in (2,4)}
        worst = {name:m.worst()[1] for name,m in (common|{"N-output":legacy[4]}).items()}
        l,u = bounds[e]
        np.testing.assert_allclose(worst["CMV-W"],u,rtol=1e-10,atol=1e-10)
        results["worst"][e] = dict(L=l,U=u,gap_pct=100*(u/l-1),variance=worst)
        # PM-sub, DE and OUE in the original search also gave endpoint centers.
        # N-output supplied some interior centers and is retained as an anchor.
        centers = [0.,1.]
        for mechanism in (*common.values(),legacy[4]):
            center = abs(mechanism.worst()[0])
            if all(abs(center-old)>1e-10 for old in centers):
                centers.append(center)
        distributions = [neighborhood(c,.05) for c in sorted(centers)]
        results["stress"][e] = {n:max(synthetic_mse(m,d,10000)["expected_mse"] for d in distributions)
                                for n,m in common.items()}

        def categorical(method, k):
            return lambda x: categorical_moments("DE" if method == "GRR" else method,e,x,k)

        for ds, data in records.items():
            aware = {n:Moments(dict(kind="linear",**d)) for n,d in spec[ds].items()}
            L,U = avg_bounds[ds,e]
            V = integrate(aware["AAA"].variance,aware["AAA"].knots,prior[ds])
            np.testing.assert_allclose(integrate(aware["CMV-D"].variance,aware["CMV-D"].knots,prior[ds]),
                                       U,rtol=1e-8,atol=1e-9)
            results["prior"][ds,e] = dict(L=L,U=U,AAA=V,lower_AAA=100*(V/U-1),upper_CMV_D=100*(U/L-1))
            for name in GENERAL:
                fn = categorical(name,4) if name in CATEGORICAL else common[name]
                results["mse"][ds,e,"No limit",name] = expected_mean_mse(fn,data)
            for name,m in aware.items():
                results["mse"][ds,e,"With the prior",name] = expected_mean_mse(m,data)
        for b in (2,4):
            label = f"At most {b} bits"
            capped = {"SR":common["SR"],"TO":common["TO"],"N-output":legacy[b],
                      "GRR":categorical("GRR",2**b),"OUE":categorical("OUE",b)}
            for name,m in continuous_mechanisms.items():
                levels = m.tuned(b)
                capped[name] = lambda x, m=m, levels=levels: m.moments(x,levels=levels)
            for ds,data in records.items():
                for name,fn in capped.items():
                    results["mse"][ds,e,label,name] = expected_mean_mse(fn,data)
    return results


def matrix(rows, columns=EPS):
    header = ["Method / setting",*[f"{x:g}" for x in columns]]
    return "\n".join(["| "+" | ".join(header)+" |","|---|"+"---:|"*len(columns)]+
        ["| "+str(label)+" | "+" | ".join(map(str,values))+" |" for label,values in rows])


def write_tables(out, results):
    def save(number, title, *blocks):
        (out/f"table_{number:02d}.md").write_text(
            "\n\n".join([f"# Table {number}. {title}",*blocks])+"\n",encoding="utf-8")

    worst,stress,prior,mse = (results[k] for k in ("worst","stress","prior","mse"))
    save(2,"Worst-case noise variance and optimality gap",
         matrix([(label,[f"{worst[e][key]:.9g}" for e in EPS]) for label,key in
                 [("Lower bound (L)","L"),("CMV-W (U)","U"),("Optimality gap (%)","gap_pct")]]+
                [(name if name!="N-output" else "N-output (4 bits)",
                  [f"{reduction(worst[e]['U'],worst[e]['variance'][name]):.2f}" for e in EPS])
                 for name in (*BASELINES,"N-output")]))
    grid = {(float(r["epsilon"]),int(r["M"])):r for r in read_csv(DATA/"grid_bounds.csv")}
    grids = (33,65,129,257,513)
    save(3,"Optimality gap (%) by grid size",
         matrix([(f"{e:g}",[f"{100*(float(grid[e,m]['U'])/float(grid[e,m]['L'])-1):.2g}" for m in grids])
                 for e in EPS],grids),
         "Largest observed wall time across the eight budgets: " +
         ", ".join(f"M={m}: {max(float(r['wall_seconds']) for r in read_csv(DATA/'grid_times.csv') if int(r['M'])==m and r['exit_code']=='0'):.2f} s"
                   for m in (33,129)) + ".")

    def macro(e,setting,method):
        return np.mean([mse[ds,e,setting,method] for ds in DATASETS])

    general = "No limit"
    competitors = BASELINES+CATEGORICAL
    blocks = [matrix([(name,[f"{reduction(macro(e,general,'CMV-W'),macro(e,general,name)):.2f}" for e in EPS])
                      for name in competitors])]
    for name in ("CMV-W","CMV-U"):
        blocks.extend([f"{name} vs. the best existing method",
            matrix([(title,[f"{reduction(mse[ds,e,general,name],min(mse[ds,e,general,b] for b in competitors)):.2f}"
                            for e in EPS]) for ds,title in DATASETS.items()])])
    save(4,"Mean squared error reduction (%) on real datasets",*blocks)
    save(5,"Reduction (%) of maximum MSE over the shared stress distributions",
         matrix([(n,[f"{reduction(stress[e]['CMV-W'],stress[e][n]):.2f}" for e in EPS])
                 for n in (*BASELINES,"CMV-U")]))
    save(6,"Excess over optimal expected noise variance for the histogram (%)",
         matrix([(f"{title}: {label}",[f"{prior[ds,e][key]:.2g}" for e in EPS])
                 for ds,title in DATASETS.items() for label,key in
                 [("CMV-D, at most","upper_CMV_D"),("AAA, at least","lower_AAA")]]))
    caps = [f"At most {b} bits" for b in (2,4)]
    save(7,"MSE reduction (%) by N-output under a report-size limit",
         *[block for setting in caps for block in [setting,
            matrix([(n,[f"{reduction(macro(e,setting,'N-output'),macro(e,setting,n)):.2f}" for e in EPS])
                    for n in competitors])]])
    for i,(ds,title) in enumerate(DATASETS.items(),8):
        save(i,f"Expected MSE (×10⁵) on {title}",
             *[block for setting,names in [(general,GENERAL),("With the prior",("AAA","CMV-D")),
                                           *[(cap,CAPS) for cap in caps]]
               for block in [setting,matrix([(n,[f"{mse[ds,e,setting,n]*1e5:.4g}" for e in EPS]) for n in names])]])
    with (out/"expected_mse.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset","epsilon","setting","mechanism","expected_mse"])
        writer.writerows([(*k,v) for k,v in mse.items()])


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.font_manager import FontProperties, findfont
    with np.load(DATA/"figure2_cmv_w_bins.npz",allow_pickle=False) as z:
        grid,edges,cmv = z["grid"],z["edges"],z["probabilities"]
    pm = continuous("PM",3.)
    def overlap(left,right):
        return np.maximum(0,np.minimum(edges[1:],right)-np.maximum(edges[:-1],left))
    center = pm.s*grid[:,None]
    mass = pm.low*overlap(-pm.A,pm.A)+(pm.high-pm.low)*overlap(center-pm.h,center+pm.h)
    np.testing.assert_allclose(mass.sum(axis=1),1,atol=1e-12)
    plt.rcParams.update({"font.size":8,"pdf.fonttype":42})
    try:
        findfont(FontProperties(family="Times New Roman"),fallback_to_default=False)
    except ValueError:
        plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix"})
    else:
        family = "Times New Roman"
        plt.rcParams.update({"font.family":family,"mathtext.fontset":"custom",
                             "mathtext.rm":family,"mathtext.it":family+":italic",
                             "mathtext.bf":family+":bold","mathtext.cal":family})
    fig = plt.figure(figsize=(3.4,1.7))
    left = fig.add_axes([.15,.24,.28,.68])
    right = fig.add_axes([.48,.24,.28,.68],sharey=left)
    cax = fig.add_axes([.795,.24,.024,.68])
    cmap = plt.get_cmap("viridis")
    cmap = cmap.with_extremes(bad="white",under=cmap(0.))
    norm = LogNorm(vmin=1e-3,vmax=1)
    xedges = np.r_[grid[0],(grid[:-1]+grid[1:])/2,grid[-1]]
    for ax,p in [(left,cmv),(right,mass)]:
        mesh = ax.pcolormesh(xedges,edges,np.ma.masked_less(p.T,1e-6),cmap=cmap,norm=norm,rasterized=True)
        ax.set(xlabel="Input $x$",xlim=(-1,1),ylim=(-1.7,1.7),xticks=[-1,0,1],yticks=[-1.5,0,1.5])
        ax.tick_params(length=2,pad=2)
    left.set_ylabel("Report $y$",labelpad=1)
    right.tick_params(axis="y",left=False,labelleft=False)
    cb = fig.colorbar(mesh,cax=cax,ticks=10.**np.arange(-3,1),extend="min")
    cb.solids.set_rasterized(False)
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=2,pad=2)
    cb.set_label("Probability per bin",labelpad=2)
    fig.savefig(out/"figure_02.pdf",dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    write_tables(args.out,calculate())
    figure(args.out)
    print(f"Tables 2-11 and Figure 2 written to {args.out}")


if __name__ == "__main__":
    main()
