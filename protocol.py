"""
The paper's evaluation protocol (Sec. 4.1 "Protocol", Algorithm 3).

VERBATIM
  "From both Df and Dr, we then sample n in {400,1000,2000} instances
   repeatedly for m times to create m subsets from each. These 2m subsets serve
   as the evaluation targets, with known labels indicating whether each subset
   originated from the training data of h^re, i.e. {(Si,1)|Si subset Dr} and
   {(Si,0)|Si subset Df}. We then apply our method to each target subset Si to
   classify its in- or out-of-training status with Eq. 2. The classification F1
   score over the 2m subsets indicates how well our method can distinguish
   between in- and out-of-training data under the h^re."

WHY THIS REPLACES ../definitive.py's DESIGN
  definitive.py produced ONE verdict per model and scored it correct/incorrect.
  The paper's unit of measurement is an F1 over 2m subsets. A single verdict is
  a 1-sample estimate of that F1, so "24/24 correct" and "5/12 correct" are not
  comparable to any number in Table 1. Everything is now scored their way.

TWO DETAILS THE PAPER LEAVES UNSTATED  (both decided here, both recorded)

  1. WHERE THE REFERENCES COME FROM. Algorithm 2 requires S_IT and S_OOT but
     Sec. 4.1 never says how they are drawn. Only one choice is non-circular:
        S_IT  subset D_r, disjoint from every target subset
        S_OOT subset the held-out TEST set
     Drawing S_OOT from D_f would be circular -- D_f is what is being
     classified. Enforced by assertion below.

  2. REFERENCE SET SIZE. H(S,h) is a distribution over HSIC values computed on
     |S|/2 x |S|/2 kernels, so its location and spread depend on |S|. Comparing
     a size-n target against differently-sized references would confound the
     JSD with subset size. References are therefore drawn at size n, matching
     the target. (If the authors did otherwise, Table 1's |S| trend is partly a
     reference-size artefact -- worth asking them.)

PERFORMANCE
  Activations are extracted ONCE per checkpoint into an ActBank and indexed per
  subset. Extracting per subset instead would re-run the network 50x over
  overlapping data and made the full sweep infeasible.
"""

import numpy as np

import hsic

FULL_BANK_MAX_DIM = 2048       # above this, hold activations per-subset instead


class ActBank:
    """h_tap(x) for every train and test image, extracted once per checkpoint.

    At the hp tap (dim 192) this is ~92 MB for CIFAR-10 and turns each target
    subset into an array index. For wide intermediate taps (h1 = 98304) a full
    bank is many GB, so those fall back to on-demand extraction.
    """

    def __init__(self, model, tr_clean, te, tap="hp", activations_fn=None):
        if activations_fn is None:
            from pipeline import activations as activations_fn
        self.fn = activations_fn
        self.model, self.tr, self.te, self.tap = model, tr_clean, te, tap
        probe = activations_fn(model, tr_clean, np.arange(2), tap=tap)
        self.dim = probe.shape[1]
        self.full = self.dim <= FULL_BANK_MAX_DIM
        if self.full:
            self.A_tr = activations_fn(model, tr_clean, np.arange(len(tr_clean)), tap=tap)
            self.A_te = activations_fn(model, te, np.arange(len(te)), tap=tap)

    def train(self, idx):
        return self.A_tr[idx] if self.full else self.fn(self.model, self.tr, idx, tap=self.tap)

    def test(self, idx):
        return self.A_te[idx] if self.full else self.fn(self.model, self.te, idx, tap=self.tap)

    def predictions(self):
        """argmax logits for the whole train and test sets, free from the bank.

        Both architectures end `logits = head(hp)` with head a single Linear, so
        at tap='hp' this is EXACT, not an approximation -- and it removes the
        separate 60k-image accuracy pass that cost ~12 s at every checkpoint.
        """
        if self.tap != "hp" or not self.full:
            raise RuntimeError("predictions() requires the full hp bank")
        import torch
        W = self.model.head.weight.detach().cpu().numpy().astype(np.float64)
        b = self.model.head.bias.detach().cpu().numpy().astype(np.float64)
        return ((self.A_tr @ W.T + b).argmax(1), (self.A_te @ W.T + b).argmax(1))


def sample_subsets(pool, n, m, rng):
    """m subsets of size n drawn from pool (without replacement within a subset)."""
    return [rng.choice(pool, size=n, replace=False) for _ in range(m)]


def evaluate(bank, D_r, D_f, n_test, n, m=25, sigma_mode="sqrtdim",
             T=hsic.T_PERM, seed=0):
    """Full Sec. 4.1 protocol for one model at one (n, tap, sigma) setting.

    bank   : ActBank for the model/tap being evaluated
    n_test : len(test set), for drawing S_OOT
    Returns dict with f1, accuracy, otr, margins, and the Sec. 3.1 test.
    """
    rng = np.random.default_rng(5000 + seed)

    # references first, so targets can be drawn disjoint from S_IT
    ref_it = rng.choice(D_r, size=n, replace=False)
    ref_oot = rng.choice(n_test, size=n, replace=False)
    pool_r = np.setdiff1d(D_r, ref_it)

    # circularity guards (unstated detail #1)
    assert np.intersect1d(ref_it, D_f).size == 0, "S_IT leaked into D_f"
    assert np.intersect1d(pool_r, ref_it).size == 0, "target pool overlaps S_IT"

    tgt_in = sample_subsets(pool_r, n, m, rng)      # label 1, in-training
    tgt_out = sample_subsets(D_f, n, m, rng)        # label 0, out-of-training

    A_it = bank.train(ref_it)
    A_oot = bank.test(ref_oot)

    sigma = hsic.bandwidth(A_it, sigma_mode)
    H_IT, H_OOT = hsic.references(A_it, A_oot, T=T, sigma=sigma, seed=seed)

    preds, preds_pg, labels, margins = [], [], [], []
    for lab, subsets in ((1, tgt_in), (0, tgt_out)):
        for i, idx in enumerate(subsets):
            r = hsic.is_in_training(bank.train(idx), None, None, T=T,
                                    sigma=sigma, seed=seed + 7 * i,
                                    H_IT=H_IT, H_OOT=H_OOT)
            preds.append(int(r["in_training"]))
            preds_pg.append(int(r["in_training_pairgrid"]))
            labels.append(lab)
            margins.append(r["margin"])

    preds = np.array(preds)
    preds_pg = np.array(preds_pg)
    labels = np.array(labels)
    margins = np.array(margins)

    def _score(p):
        tp = int(((p == 1) & (labels == 1)).sum())
        fp = int(((p == 1) & (labels == 0)).sum())
        fn = int(((p == 0) & (labels == 1)).sum())
        tn = int(((p == 0) & (labels == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
                    f1=2 * prec * rec / max(prec + rec, 1e-12),
                    accuracy=float((p == labels).mean()),
                    otr=float((p[labels == 0] == 0).mean()),
                    in_training_rate_on_Dr=float((p[labels == 1] == 1).mean()))

    s = _score(preds)
    s_pg = _score(preds_pg)
    sec31 = hsic.section31_test(A_it, A_oot, T=T, sigma=sigma, seed=seed)

    return dict(
        **{k: float(v) if isinstance(v, float) else v for k, v in s.items()},
        f1_pairgrid=float(s_pg["f1"]), accuracy_pairgrid=float(s_pg["accuracy"]),
        otr_pairgrid=float(s_pg["otr"]),
        grid_disagreement=float((preds != preds_pg).mean()),
        mean_margin_Dr=float(margins[:m].mean()),
        mean_margin_Df=float(margins[m:].mean()),
        sigma=float(sigma), n=n, m=m, tap=bank.tap, sigma_mode=sigma_mode,
        sec31_p=sec31["p"], sec31_effect=sec31["effect"],
        sec31_detected=sec31["detected"],
        sec31_median_IT=sec31["median_IT"], sec31_median_OOT=sec31["median_OOT"],
    )
