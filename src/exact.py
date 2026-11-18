"""Small independently reusable rational arithmetic and certificate I/O."""
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path


def rational(value, digits=16):
    return F(format(float(value), f".{digits}g"))


def exp_enclosure(epsilon, digits=40):
    e = F(str(epsilon))
    if not 0 < e <= 100:
        raise ValueError("supported epsilon range: (0,100]")
    n = max(120, int(e)*4+120)
    term = total = F(1)
    for k in range(1, n+1):
        term *= e/k
        total += term
    first = term*e/(n+1)
    upper = total + first/(1-e/(n+2))
    den = 10**digits
    lo = F((total*den).__floor__(), den)
    hi = F((upper*den).__ceil__(), den)
    if lo <= 1:
        raise ValueError("epsilon too small for certificate precision")
    return lo, hi


def outward(value, up):
    result = float(value)
    direction = math.inf if up else -math.inf
    while (F.from_float(result) < value if up else F.from_float(result) > value):
        result = math.nextafter(result, direction)
    return result


def seal(data):
    data = dict(data)
    data.pop("hash", None)
    data["hash"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return data


def check_hash(data):
    if seal(data)["hash"] != data.get("hash"):
        raise ValueError("certificate hash mismatch")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def quadratic_min(a, b, c, lo=None, hi=None):
    """Minimum of a*y^2+b*y+c on a closed interval; a>0 for tails."""
    points = [v for v in (lo, hi) if v is not None]
    if a > 0:
        vertex = -b/(2*a)
        if (lo is None or lo <= vertex) and (hi is None or vertex <= hi):
            points.append(vertex)
    elif lo is None or hi is None:
        raise ValueError("noncoercive tail passed to quadratic_min")
    return min(a*y*y+b*y+c for y in points)


def sqrt_enclosure(value, digits=24):
    if value < 0:
        raise ValueError("negative radicand")
    den = 10**digits
    n = math.isqrt(value.numerator*den*den//value.denominator)
    lo = F(n, den)
    return lo, lo if lo*lo == value else F(n+1, den)
