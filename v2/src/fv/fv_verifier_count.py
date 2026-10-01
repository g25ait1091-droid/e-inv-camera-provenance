"""Count the checks of the repository verifier (RESULTS.md Entry 116 item 6e, re-run after verify_v2.py was extended
with the Entry 114-119 checks on 1 Oct 2026).

Runs `python verify_v2.py` in the local repository clone (CPU, no downloads; the clone's working tree, which is
not committed or pushed), parses its per-check lines and its "N of M checks agree" line, and records the count,
the SHA-256 of the verifier and the clone's HEAD. Writes out/fv_verifier_count.json (never overwritten).
Run:  python src/fv/fv_verifier_count.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = EINV.REPO
DST = (EINV.V2 + "/out/fv_verifier_count.json")
GIT = ("git",)


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_head():
    for g in GIT:
        try:
            r = subprocess.run([g, "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout.strip()
        except OSError:
            continue
    return None


def main():
    if os.path.exists(DST):
        raise SystemExit(f"{DST} exists; result files are never overwritten")
    t0 = time.time()
    r = subprocess.run([sys.executable, "verify_v2.py"], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = r.stdout.splitlines()
    ok = sum(1 for l in lines if l.startswith("OK "))
    bad = [l for l in lines if l.startswith("DISAGREE")]
    m = [re.match(r"(\d+) of (\d+) checks agree", l) for l in lines]
    m = [x for x in m if x]
    assert m, r.stdout[-2000:] + r.stderr[-2000:]
    agree, executed = int(m[-1].group(1)), int(m[-1].group(2))
    assert agree == ok and executed == ok + len(bad), (agree, executed, ok, len(bad))
    sections = sorted({int(x.group(1)) for x in re.finditer(r"^# (\d+)\. ", open(os.path.join(REPO, "verify_v2.py"),
                                                                                    encoding="utf-8").read(), re.M)})
    res = {"entry": "RESULTS.md Entry 116 item 6e (count re-run after the verifier was extended, 1 Oct 2026)",
           "script": "src/fv/fv_verifier_count.py",
           "verifier": {"file": "verify_v2.py (repository clone, working tree; not committed or pushed)",
                        "sha256": sha(os.path.join(REPO, "verify_v2.py")), "clone_HEAD": git_head(),
                        "exit_code": r.returncode, "checks_executed": executed, "checks_agreeing": agree,
                        "disagreeing": bad, "numbered_sections": sections,
                        "entry_114_119_checks": sum(1 for l in lines if re.match(r"(OK|DISAGREE)\s+E11[4-9] ", l)),
                        "summary_line": m[-1].group(0)},
           "stdout": lines, "runtime_s": time.time() - t0}
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"{agree} of {executed} checks agree (Entry 114-119 checks: {res['verifier']['entry_114_119_checks']}); "
          f"exit {r.returncode}; wrote {DST}")


if __name__ == "__main__":
    main()
