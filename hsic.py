"""
SDE core, faithful to Zhang et al. ICLR 2026 and fast enough for the full
replication (Algorithms 1-3, Appendix C).

FAITHFULNESS (each line traced to the paper)
  Kernel      Gaussian RBF, k(x,x') = exp(-||x-x'||^2 / 2 sigma^2)      Eq. 4
  Estimator   BIASED, HSIC = 1/(n-1)^2 * Tr(K H L H)                    Sec. 2.2
  Bandwidth   DEFAULT sigma = sqrt(dim)                                 Sec. 4.1.1
              (sigma = 128 for logits of 10-class tasks, App. D)
              median heuristic supported as their second heuristic
  Alg 1       fixed split-half, T permutations of S2 -> H(S,h)          App. C
  Alg 2       is_in_training: JSD(H_tar,H_IT) < JSD(H_tar,H_OOT)        Eq. 2/3
  Alg 3       unlearn_eval: OOT rate over m target subsets              App. C

WHAT CHANGED VS ../sde.py, AND WHY IT IS NOT A CHANGE OF METHOD
  1. DEFAULT BANDWIDTH. ../sde.py defaulted to the median heuristic. Sec. 4.1.1
     ends: "unless otherwise specified, we use sigma = sqrt(dim) as the default
     choice." The median heuristic is their *comparison* heuristic. Running the
     replication on the median default would not be their method.

  2. O(n^2) PERMUTATIONS instead of O(n^3). Using the cyclic property,
         Tr(K H L H) = Tr((H K H) L) = sum_ij Kc[i,j] * L[i,j],   Kc = H K H
     so double-centre K ONCE and each of the T permutations is an elementwise
     sum against the permuted L. Numerically identical (asserted in selftest);
     ~1000x faster at n=1000. Without this the full Table 1 sweep is infeasible.
     (../sde.py already fixed the related 200x waste of recomputing the kernel;
     this removes the remaining matmuls.)

  3. MEMORY-SAFE distances. ../sde.py built an (n,n,d) intermediate: at n=1000,
     d=192 that is 1.5 GB, and at the h1 tap (d=98304) it is impossible. Uses
     the ||x||^2 + ||y||^2 - 2<x,y> form instead.

  All three are implementation, not method. reference_hsic_biased() is kept as
  the naive O(n^3) transcription of Sec. 2.2 and the selftest asserts equality.
"""

import numpy as np
from scipy.stats import mannwhitneyu

T_PERM = 200          # permutation times T (Alg. 1); ../sde.py used 200
JSD_BINS = 50


# ------------------------------------------------------------------ kernels

def sqdist(X):
    """Pairwise squared Euclidean distances without an (n,n,d) intermediate."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def bandwidth(X, mode="sqrtdim", n_cls=None):
    """The paper's two heuristics (Sec. 4.1.1).

    'sqrtdim' : sigma = sqrt(dim), their stated default. App. D overrides this
                to sigma = 128 for the logit tap of 10-class tasks, where
                sqrt(dim) ~= 3 is "too small".
    'median'  : median pairwise Euclidean distance, their comparison heuristic.
    """
    d = X.shape[1]
    if mode == "sqrtdim":
        if d <= 16:                      # the logit tap; App. D special case
            return 128.0
        return float(np.sqrt(d))
    if mode == "median":
        n = min(len(X), 1000)
        idx = np.random.default_rng(0).choice(len(X), n, replace=False)
        d2 = sqdist(X[idx])
        med = float(np.median(np.sqrt(d2[np.triu_indices(n, 1)])))
        return med if med > 0 else 1.0
    raise ValueError(mode)


def rbf(X, sigma):
    return np.exp(-sqdist(X) / (2.0 * sigma ** 2))


# ------------------------------------------------------------------ HSIC

def _centre(K):
    """H K H with H = I - 11^T/n, computed without forming H."""
    rm = K.mean(0, keepdims=True)
    cm = K.mean(1, keepdims=True)
    return K - rm - cm + K.mean()


def reference_hsic_biased(K, L):
    """Naive transcription of Sec. 2.2: 1/(n-1)^2 Tr(K H L H). O(n^3).
    Kept only so the fast path can be asserted equal to it."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / ((n - 1) ** 2))


def hsic_biased_from_centred(Kc, L):
    """Tr(K H L H) = sum_ij Kc[i,j] L[i,j] with Kc = H K H. O(n^2)."""
    n = Kc.shape[0]
    return float(np.sum(Kc * L) / ((n - 1) ** 2))


# ------------------------------------------------------- Algorithm 1

def estimate_hsic_distribution(A, T=T_PERM, sigma=None, sigma_mode="sqrtdim",
                               seed=0):
    """Algorithm 1 (App. C). Returns T HSIC values -> the distribution H(S,h).

        S1 = S[:len(S)//2];  S2 = S[len(S)//2:]
        for _ in range(T):  S2 = RandomShuffle(S2); H.append(HSIC(h(S1),h(S2)))

    Shuffling the ROWS of S2 permutes its kernel matrix and does not change the
    kernel values, so both kernels are computed once and L is permuted.
    """
    A = np.asarray(A, dtype=np.float64)
    if sigma is None:
        sigma = bandwidth(A, sigma_mode)
    n = len(A) // 2 * 2
    A = A[:n]
    half = n // 2
    S1, S2 = A[:half], A[half:]              # Alg. 1 splits in order, not at random

    Kc = _centre(rbf(S1, sigma))
    L = rbf(S2, sigma)
    rng = np.random.default_rng(seed)
    out = np.empty(T)
    for i in range(T):
        q = rng.permutation(half)
        out[i] = hsic_biased_from_centred(Kc, L[np.ix_(q, q)])
    return out


# ------------------------------------------------------- Eq. 3 / Algorithm 2

def _edges(sample_sets, bins):
    lo = min(float(s.min()) for s in sample_sets)
    hi = max(float(s.max()) for s in sample_sets)
    if hi <= lo:
        return None
    return np.linspace(lo, hi, bins + 1)


def jsd(p_samples, q_samples, bins=JSD_BINS, edges=None):
    """Jensen-Shannon divergence between two empirical distributions (Eq. 3).

    `edges` lets the caller impose a grid shared with other comparisons; see
    is_in_training. Without it the grid spans this pair's pooled support.
    """
    if edges is None:
        edges = _edges([p_samples, q_samples], bins)
        if edges is None:
            return 0.0
    p = np.histogram(p_samples, bins=edges)[0].astype(float)
    q = np.histogram(q_samples, bins=edges)[0].astype(float)
    p /= max(p.sum(), 1.0)
    q /= max(q.sum(), 1.0)
    m = 0.5 * (p + q)

    def _kl(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log2(a[k] / np.maximum(b[k], 1e-12))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def is_in_training(A_tar, A_IT, A_OOT, T=T_PERM, sigma=None,
                   sigma_mode="sqrtdim", seed=0, H_IT=None, H_OOT=None,
                   bins=JSD_BINS):
    """Algorithm 2 / Eq. 2. ['in_training'] is the verdict.

    H_IT/H_OOT may be passed in precomputed: the reference distributions do not
    depend on the target, and Alg. 3 evaluates m targets against the same pair.

    ---------------------------------------------------------------------------
    AN UNSTATED DETAIL THAT AFFECTS THE VERDICT: THE JSD GRID

    Eq. 2 compares two divergences, JSD(H_tar,H_IT) vs JSD(H_tar,H_OOT). JSD on
    *samples* requires a discretisation, and the paper specifies none.

    Computing each divergence on its own pair's pooled support -- the obvious
    reading, and what this file did initially -- makes the two numbers
    incommensurable. If H_IT is widely spread and H_OOT narrow, the IT
    comparison is binned coarsely and the OOT comparison finely, and finer bins
    generally yield LARGER JSD. So the criterion acquires a bias that depends on
    the relative spread of the two references rather than on the target.

    Both are therefore computed:
      in_training           shared grid over all THREE distributions (default,
                            and the only one under which the two divergences are
                            comparable quantities)
      in_training_pairgrid  per-pair grid, the naive reading

    Carrying both costs microseconds -- the H distributions are already in hand
    -- and disagreement between them is itself a result about the method.
    ---------------------------------------------------------------------------
    """
    H_tar = estimate_hsic_distribution(A_tar, T, sigma, sigma_mode, seed)
    if H_IT is None:
        H_IT = estimate_hsic_distribution(A_IT, T, sigma, sigma_mode, seed + 1)
    if H_OOT is None:
        H_OOT = estimate_hsic_distribution(A_OOT, T, sigma, sigma_mode, seed + 2)

    e = _edges([H_tar, H_IT, H_OOT], bins)
    d_it = jsd(H_tar, H_IT, bins, edges=e)
    d_oot = jsd(H_tar, H_OOT, bins, edges=e)
    d_it_p = jsd(H_tar, H_IT, bins)
    d_oot_p = jsd(H_tar, H_OOT, bins)
    return {
        "in_training": bool(d_it < d_oot),          # Eq. 2, shared grid
        "in_training_pairgrid": bool(d_it_p < d_oot_p),
        "d_to_IT": d_it,
        "d_to_OOT": d_oot,
        "margin": float(d_oot - d_it),              # >0 => judged in-training
        "margin_pairgrid": float(d_oot_p - d_it_p),
        "med_tar": float(np.median(H_tar)),
        "med_IT": float(np.median(H_IT)),
        "med_OOT": float(np.median(H_OOT)),
    }


def references(A_IT, A_OOT, T=T_PERM, sigma=None, sigma_mode="sqrtdim", seed=0):
    """Precompute the two reference distributions for a batch of targets."""
    return (estimate_hsic_distribution(A_IT, T, sigma, sigma_mode, seed + 1),
            estimate_hsic_distribution(A_OOT, T, sigma, sigma_mode, seed + 2))


# --------------------------------------------- Sec. 3.1 motivating claim

def section31_test(A_IT, A_OOT, T=T_PERM, sigma=None, sigma_mode="sqrtdim",
                   seed=0):
    """The paper's MOTIVATING observation, not its criterion: one-sided
    Mann-Whitney for H(S_IT) > H(S_OOT), rejected at p < 0.01.

    Finding A of this project is that this ordering is reversed (p = 1.0)
    precisely where Eq. 2 works perfectly, so it is tracked alongside every
    Eq. 2 verdict rather than discarded."""
    h_in = estimate_hsic_distribution(A_IT, T, sigma, sigma_mode, seed)
    h_out = estimate_hsic_distribution(A_OOT, T, sigma, sigma_mode, seed + 1)
    U, p = mannwhitneyu(h_in, h_out, alternative="greater")
    return {
        "p": float(p),
        "effect": float(2.0 * U / (len(h_in) * len(h_out)) - 1.0),
        "median_IT": float(np.median(h_in)),
        "median_OOT": float(np.median(h_out)),
        "detected": bool(p < 0.01),
    }


# ------------------------------------------------------------------ selftest

def selftest(verbose=True):
    """Falsification before method. Four checks, all must pass."""
    rng = np.random.default_rng(0)
    ok = True

    # 1. fast path == naive Sec. 2.2 transcription
    X, Y = rng.normal(size=(120, 8)), rng.normal(size=(120, 8))
    K, L = rbf(X, 2.0), rbf(Y, 2.0)
    a = reference_hsic_biased(K, L)
    b = hsic_biased_from_centred(_centre(K), L)
    same = np.isclose(a, b, rtol=1e-9, atol=1e-14)
    ok &= same
    if verbose:
        print(f"  [1] fast O(n^2) == naive O(n^3) Tr(KHLH): "
              f"{a:.12e} vs {b:.12e}  {'OK' if same else 'MISMATCH'}")

    # 2. biased estimator is non-negative under permutation (its whole point;
    #    negative medians were how the unbiased-estimator bug was caught before)
    H = estimate_hsic_distribution(rng.normal(size=(400, 16)), T=100)
    pos = bool((H > 0).all())
    ok &= pos
    if verbose:
        print(f"  [2] biased estimator > 0 under permutation: median={np.median(H):.3e}"
              f"  {'OK' if pos else 'FAIL'}")

    # 3. NEGATIVE CONTROL, must fail to detect: no structure anywhere.
    r = is_in_training(rng.normal(size=(400, 16)),
                       rng.normal(size=(400, 16)),
                       rng.normal(size=(400, 16)))
    if verbose:
        print(f"  [3] null control (all i.i.d. Gaussian): margin={r['margin']:+.4f}"
              f"  -> verdict is arbitrary by construction, margin must be small")
    small = abs(r["margin"]) < 0.5
    ok &= small

    # 4. POSITIVE CONTROL: target drawn from the same population as the IT
    #    reference must be judged in-training.
    shared = rng.normal(size=(1, 16))
    pop_it = lambda n: rng.normal(size=(n, 16)) + 1.5 * shared
    r2 = is_in_training(pop_it(400), pop_it(400), rng.normal(size=(400, 16)))
    ok &= r2["in_training"]
    if verbose:
        print(f"  [4] positive control (target ~ IT population): "
              f"in_training={r2['in_training']} margin={r2['margin']:+.4f}"
              f"  {'OK' if r2['in_training'] else 'FAIL'}")

    if verbose:
        print(f"\n  HSIC SELFTEST {'PASS' if ok else 'FAIL - do not use downstream'}")
    return bool(ok)


if __name__ == "__main__":
    print("=" * 74)
    print("hsic.py selftest")
    print("=" * 74)
    selftest()
