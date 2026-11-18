"""Construct an exactly feasible nearby channel; independently audit its upper bound.

The repaired channel is explicit and separate from the raw solver output. A tiny
input-independent mixture repairs privacy. Two extra outputs repair the means
exactly. Output count is unrestricted, so these extra outputs are permitted and
always included in N_effective. No tolerance-based feasibility is called exact.
"""
import json
import math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from .exact import rational, exp_enclosure, outward, seal, check_hash, save_json
from .channel import GridChannel


def exact_channel_audit(epsilon, grid, outputs, p):
    m, k = len(grid), len(outputs)
    if m < 2 or k < 1 or len(p) != m or any(len(row) != k for row in p):
        raise ValueError("invalid channel shape")
    if grid[0] != -1 or grid[-1] != 1 or any(a >= b for a,b in zip(grid,grid[1:])):
        raise ValueError("invalid grid")
    E = exp_enclosure(epsilon)[0]
    for x, row in zip(grid, p):
        if min(row) < 0 or sum(row) != 1 or sum(a*b for a,b in zip(outputs,row)) != x:
            raise ValueError("exact normalization/nonnegativity/unbiasedness failed")
    for col in zip(*p):
        if max(col) > E*min(col):
            raise ValueError("exact LDP failed")
    second = [sum(y*y*prob for y,prob in zip(outputs,row)) for row in p]
    grid_values = [s-x*x for s,x in zip(second,grid)]
    candidates = []
    for j in range(m-1):
        slope = (second[j+1]-second[j])/(grid[j+1]-grid[j])
        star = min(grid[j+1],max(grid[j],slope/2))
        val = second[j]+slope*(star-grid[j])-star*star
        candidates.append((val,star))
    U, where = max(candidates)
    Ug, delta = max(grid_values), max(b-a for a,b in zip(grid,grid[1:]))
    if not Ug <= U <= Ug+delta*delta/4:
        raise ValueError("interpolation bound failed")
    return dict(U_cont=U, U_grid=Ug, maximizer=where, Delta=delta)


def certify_continuous(channel, epsilon, path=None, symmetry=False):
    x = [F(str(v)) for v in channel.grid]
    y = [rational(v) for v in channel.outputs]
    p = [[max(F(0),rational(v)) for v in row] for row in channel.P_grid]
    totals = [sum(row) for row in p]
    if any(total <= 0 for total in totals):
        raise ValueError("empty probability row")
    p = [[v/total for v in row] for row,total in zip(p,totals)]
    k = len(y)
    E = exp_enclosure(epsilon)[0]
    mixing = F(0)
    for col in zip(*p):
        excess = max(col)-E*min(col)
        if excess > 0:
            mixing = max(mixing, excess/(excess+(E-1)/k))
    if mixing:
        mixing = min(F(1),F((mixing*10**16).__ceil__()+1,10**16))
        p = [[(1-mixing)*v+mixing/k for v in row] for row in p]
    means = [sum(a*b for a,b in zip(y,row)) for row in p]
    errors = [t-m for t,m in zip(x,means)]
    error = max(abs(d) for d in errors)
    delta, amplitude = F(0), F(0)
    if error:
        kappa = (E-1)/(E+1)
        amplitude = F(max(1, math.ceil(1/math.tanh(float(epsilon)/2))))
        delta = max(F(1,10**14), 2*error/(amplitude*kappa))
        delta = F((delta*10**16).__ceil__()+1,10**16)
        if delta >= F(1,4):
            raise ValueError("raw channel needs excessive mean repair; not certified")
        y = [a/(1-delta) for a in y] + [-amplitude,amplitude]
        p = [[(1-delta)*v for v in row]+[delta/2-d/(2*amplitude),delta/2+d/(2*amplitude)]
             for row,d in zip(p,errors)]
    # Exact identical-value merge, including their probabilities.
    unique = sorted(set(y))
    index = {a:j for j,a in enumerate(unique)}
    merged = [[F(0)]*len(unique) for _ in x]
    for col,a in enumerate(y):
        for row in range(len(x)):
            merged[row][index[a]] += p[row][col]
    y,p = unique,merged
    if symmetry:
        if x != [-v for v in x[::-1]]:
            raise ValueError("exact channel symmetry requires a symmetric grid")
        # Exact reflection mixture preserves LDP and unbiasedness and cannot
        # increase worst variance. Merge identical +/- outputs without tolerance.
        symmetric_y = sorted(set(y) | {-v for v in y})
        old_index = {v: j for j, v in enumerate(y)}
        p = [[((row[old_index[v]] if v in old_index else F(0)) +
               (p[len(x)-1-i][old_index[-v]] if -v in old_index else F(0)))/2
              for v in symmetric_y] for i, row in enumerate(p)]
        y = symmetric_y
    audit = exact_channel_audit(epsilon,x,y,p)
    data = seal(dict(kind="continuous", epsilon=str(epsilon), grid=[str(v) for v in x],
                     outputs=[str(v) for v in y], P_grid=[[str(v) for v in row] for row in p],
                     U_grid_rational=str(audit["U_grid"]), U_cont_rational=str(audit["U_cont"]),
                     U_grid_cert=outward(audit["U_grid"],True), U_cont_cert=outward(audit["U_cont"],True),
                     maximizer=str(audit["maximizer"]), N_effective=len(y),
                     privacy_mix=str(mixing), mean_repair_mass=str(delta), repair_output_amplitude=str(amplitude),
                     raw_N_effective=len(channel.outputs), verify_status="passed",
                     reflection_symmetric=symmetry,
                     construction="rational row normalization; uniform privacy mixture; exact two-output mean repair; linear interpolation"))
    verify_continuous(data)
    if path:
        save_json(path,data)
    return data


def verify_continuous(certificate):
    data = json.loads(Path(certificate).read_text(encoding="utf-8")) if isinstance(certificate,(str,Path)) else certificate
    check_hash(data)
    if data["kind"] != "continuous":
        raise ValueError("wrong certificate type")
    x,y = [[F(v) for v in data[key]] for key in ("grid","outputs")]
    p = [[F(v) for v in row] for row in data["P_grid"]]
    audit = exact_channel_audit(data["epsilon"],x,y,p)
    if data.get("reflection_symmetric"):
        if x != [-v for v in x[::-1]] or y != [-v for v in y[::-1]] or p != [row[::-1] for row in p[::-1]]:
            raise ValueError("claimed reflection symmetry failed exact verification")
    for key in ("U_grid","U_cont"):
        if F(data[key+"_rational"]) != audit[key] or F.from_float(data[key+"_cert"]) < audit[key]:
            raise ValueError("invalid upper bound or outward rounding")
    if data["N_effective"] != len(y) or F(data["maximizer"]) != audit["maximizer"]:
        raise ValueError("inconsistent channel metadata")
    return True


def channel_from_certificate(data):
    return GridChannel([float(F(v)) for v in data["grid"]], [float(F(v)) for v in data["outputs"]],
                       [[float(F(v)) for v in row] for row in data["P_grid"]])
