"""Coverage of the three upper-limit constructions at k=6 adapter clusters.

Reviewer finding 1.2: reporting the largest of three procedures is conservative in
magnitude but says nothing about frequentist coverage. This measures it.

Design. Under a true device-transfer parameter theta_true, draw k cluster means per arm,
draw the denominator from its own sampling distribution, build each limit, and count how
often the limit covers theta_true/R. A valid 99% one-sided upper limit should cover at
least 99% of the time.

Three cluster distributions are tested, because the observed arm A means contain an
outlier (1.03e-04 against a spread of ~2e-05) that argues against normality.
"""
import numpy as np
from scipy import stats as sps

A = np.array([8.850071894e-06, -6.325359893e-06, 2.2739481612e-06,
              -2.264346126e-05, 7.531645252e-06, 1.03266094242e-04])
B = np.array([-3.41837958e-05, 1.3627266574e-05, 4.9241766654e-05,
              4.0683344753e-05, -1.6195523868e-05, 3.3124104778e-05])
R, R_SE = 3.56703416571125e-02, 1.586e-03
K, ALPHA, NSIM, NBOOT = 6, 0.01, 4000, 800
rng = np.random.default_rng(0)
sdA, sdB = A.std(ddof=1), B.std(ddof=1)
tcrit = float(sps.t.ppf(1-ALPHA/2, df=K-1))
zlo   = float(sps.norm.ppf(1-ALPHA))

print(f"observed: sd_A {sdA:.3e}  sd_B {sdB:.3e}  R {R:.4e} (SE {R_SE:.3e})")
print(f"normality of the observed cluster means — Shapiro p: "
      f"A {sps.shapiro(A).pvalue:.3f}, B {sps.shapiro(B).pvalue:.3f}")
print(f"arm A excess kurtosis {sps.kurtosis(A):+.2f}  (normal 0)\n")

def draw(dist, sd, n):
    if dist == "normal": return rng.normal(0, sd, n)
    if dist == "t3":
        x = rng.standard_t(3, n); return x/np.sqrt(3.0)*sd
    if dist == "empirical":                      # resample the centred observed means
        pool = np.r_[A-A.mean(), B-B.mean()]
        return rng.choice(pool, n, replace=True)

def bca_upper(cl, denom_draws, level):
    """BCa upper limit on the ratio, resampling clusters then the denominator."""
    obs = cl.mean()/denom_draws[0]
    bs = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = rng.integers(0, len(cl), len(cl))
        bs[b] = cl[idx].mean()/denom_draws[rng.integers(0, len(denom_draws))]
    z0 = sps.norm.ppf(np.clip((bs < obs).mean(), 1e-6, 1-1e-6))
    jk = np.array([np.delete(cl, i).mean()/denom_draws[0] for i in range(len(cl))])
    jm = jk.mean(); num = ((jm-jk)**3).sum(); den = 6*(((jm-jk)**2).sum()**1.5)
    a = num/den if den > 0 else 0.0
    z = sps.norm.ppf(level)
    q = sps.norm.cdf(z0 + (z0+z)/max(1-a*(z0+z), 1e-9))
    return float(np.quantile(bs, np.clip(q, 0, 1)))

print(f"{'dist':11s}{'theta_true':>12s}{'plug-in':>10s}{'conserv':>10s}{'BCa':>10s}"
      f"{'max-of-3':>10s}   (target 99.0%)")
print("-"*76)
rows = []
for dist in ("normal", "t3", "empirical"):
    for th in (0.0, 1.0e-05, 3.0e-05):
        cov = {k: 0 for k in ("plug", "cons", "bca", "max")}
        truth = th/R
        for _ in range(NSIM):
            cA = th + draw(dist, sdA, K); cB = th + draw(dist, sdB, K)
            Rd = np.abs(rng.normal(R, R_SE, 40)); Rd[0] = max(rng.normal(R, R_SE), 1e-6)
            Rlo = max(Rd[0] - zlo*R_SE, 1e-6)
            U = max(cA.mean() + tcrit*cA.std(ddof=1)/np.sqrt(K),
                    cB.mean() + tcrit*cB.std(ddof=1)/np.sqrt(K))
            arm = cA if (cA.mean()+tcrit*cA.std(ddof=1)/np.sqrt(K)) >= \
                        (cB.mean()+tcrit*cB.std(ddof=1)/np.sqrt(K)) else cB
            lim = {"plug": U/Rd[0], "cons": U/Rlo,
                   "bca": bca_upper(arm, Rd, 1-ALPHA/2)}
            lim["max"] = max(lim.values())
            for k_ in cov: cov[k_] += (lim[k_] >= truth)
        r = {k: 100*v/NSIM for k, v in cov.items()}
        rows.append((dist, th, r))
        flag = "" if min(r.values()) >= 98.0 else "   <- UNDERCOVERS"
        print(f"{dist:11s}{th:12.1e}{r['plug']:9.1f}%{r['cons']:9.1f}%{r['bca']:9.1f}%"
              f"{r['max']:9.1f}%{flag}", flush=True)     # each scenario takes ~a minute; show progress
                                                         # even when stdout is a pipe or a file

print("\n" + "="*76)
worst = {k: min(r[k] for _,_,r in rows) for k in ("plug","cons","bca","max")}
for k, name in (("plug","plug-in t"), ("cons","conservative denominator"),
                ("bca","BCa"), ("max","max of the three")):
    v = worst[k]
    print(f"  {name:26s} worst-case coverage {v:5.1f}%   "
          f"{'VALID at 99%' if v>=99 else 'UNDERCOVERS' if v<98 else 'marginal'}")
print("="*76)
