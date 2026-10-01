"""Shared helpers for the fv manuscript's number macros.

Every part module (src/fv/num_*.py) builds its macros with these helpers so that formatting is uniform.
A macro is a dict: name, value (full precision), text (as printed in LaTeX), source (file), key (JSON key
path or computation), entry (RESULTS.md entry, e.g. "107"), check (how the value was verified).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import json
import math
import os
import re

V2 = EINV.V2
OUT = V2 + "/out"
LEDGER = (EINV.REPO + "/analysis/FINAL_LEDGER.json")
R_REAL = 0.0356703416571125          # E2 real-image device contrast, 40 held-out photographs per body

PREFIXES = ("Data", "AE", "Obj", "Map", "Known", "Lim", "Main", "Det", "Att", "Dose", "Gen", "Pone",
            "Est", "Wt", "Shift", "Mem", "Vone")
NAME_RE = re.compile(r"^n(" + "|".join(PREFIXES) + r")[A-Z][A-Za-z]*$")


def load(path):
    """Load a JSON result file; relative paths are taken under v2/out."""
    p = path if os.path.isabs(path) or path.startswith(("D:", "C:", "G:")) else os.path.join(OUT, path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get(d, keypath):
    """Walk a dotted key path; integer components index lists."""
    cur = d
    for k in keypath.split("."):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur


def _minus(s):
    return "$-$" + s[1:] if s.startswith("-") else s


def sig(x, n=3):
    """x to n significant figures, fixed notation, trailing zeros kept; proper minus sign."""
    if x == 0:
        return "0"
    d = n - int(math.floor(math.log10(abs(x)))) - 1
    d = max(d, 0)
    s = f"{round(x, d):.{d}f}"
    return _minus(s)


def dec(x, d=3):
    """x to d decimals, proper minus sign."""
    return _minus(f"{x:.{d}f}")


def sci(x, n=3):
    """Scientific notation in math mode: $5.38\\times10^{-5}$."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    ms = f"{m:.{n - 1}f}"
    if ms.startswith("10"):
        e += 1
        ms = f"{m / 10:.{n - 1}f}"
    ms = ms.replace("-", "-")
    return f"${ms}\\times10^{{{e}}}$".replace("$-", "$-")


def integer(n):
    """Integer with a thin-space thousands separator for >= 10000 (IEEE style: 4800, 12{,}000)."""
    n = int(round(n))
    s = f"{abs(n):,}".replace(",", "{,}") if abs(n) >= 10000 else str(abs(n))
    return ("$-$" if n < 0 else "") + s


def pct_of_rreal(x, n=3, r=R_REAL):
    """A raw contrast (NCC units) as a percentage of R_real, n significant figures."""
    return sig(100.0 * x / r, n)


def macro(name, value, text, source, key, entry, check=""):
    assert NAME_RE.match(name), f"bad macro name {name!r}: n + prefix {PREFIXES} + CamelCase letters only"
    assert text not in (None, ""), name
    return dict(name=name, value=value, text=text, source=source, key=key, entry=str(entry), check=check)
