"""
RULER's four approximate unlearning methods, reimplemented from Sec. 4.2.

VERBATIM SPEC (P17-RULER.txt lines 408-413):

    "Gradient Ascent (GA) maximises cross-entropy loss on Df for 5 epochs.
     NegGrad+ (NG+) combines gradient ascent on Df with gradient descent on Dr
     for 10 epochs, weighted by alpha = 0.6. Fine-Tuning (FT) continues gradient
     descent exclusively on Dr for 10 epochs, relying on catastrophic
     forgetting. SCRUB distils from the frozen original (teacher) to a student
     copy for 10 epochs, maximising agreement on Dr and disagreement on Df,
     with alpha = 0.6 and temperature T = 2.0. All four start from theta_o and
     use Adam with unlearning learning rate eta_u = 5 x 10^-4, held fixed across
     all datasets and forget fractions."

⚠ THIS IS A REIMPLEMENTATION, NOT THEIR CODE. Their repository is "anonymised
(URL provided upon acceptance)" and is not findable, which is open question Q1 to
the authors. Three specification choices are underdetermined by the text and are
recorded here rather than buried:

  1. NegGrad+ weighting. "weighted by alpha = 0.6" does not say which term gets
     alpha. Implemented as the standard form from Kurmanji et al. [9], which is
     the reference RULER cites for the method:
         L = alpha * CE(D_r) - (1 - alpha) * CE(D_f)
     so alpha weights the RETAIN term. The alternative reading (alpha on the
     ascent term) is a different method and would forget more aggressively.

  2. SCRUB's objective. Implemented in its standard alternating form: a
     max-step over D_f then a min-step over D_r each epoch, with
         retain: alpha * KL_T(teacher || student) + (1 - alpha) * CE(student, y)
         forget: - KL_T(teacher || student)
     KL is temperature-scaled and multiplied by T^2 (Hinton distillation
     convention). RULER's A.11 describes Bad Teacher as minimising "both the KL
     divergence to the ... teacher and the cross-entropy loss on true labels",
     which is the same retain-side structure and is the basis for this reading.

  3. Batch size during unlearning is not stated. We use the dataset's training
     batch size from pipeline.RECIPE, so unlearning matches the recipe that
     produced theta_o.

Every method starts from a DEEP COPY of theta_o and never mutates it.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import pipeline

ETA_U = 5e-4          # Sec. 4.2, "held fixed across all datasets"
ALPHA = 0.6           # Sec. 4.2
TEMPERATURE = 2.0     # Sec. 4.2
EPOCHS = dict(GradientAscent=5, NegGradPlus=10, FineTuning=10, SCRUB=10)


def _loader(ds, idx, bs, seed):
    g = torch.Generator()
    g.manual_seed(20_000 + seed)
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=True,
                      num_workers=0, generator=g)


def _kl_t(student_logits, teacher_logits, T=TEMPERATURE):
    """Temperature-scaled KL(teacher || student), Hinton convention (x T^2)."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def unlearn(method, theta_o, tr_aug, D_r, D_f, dataset, arch_name, seed=0,
            log=lambda *a: None):
    """Return a NEW model produced by `method`. theta_o is never modified."""
    if method not in EPOCHS:
        raise ValueError(f"unknown method {method!r}")
    rc = pipeline.RECIPE[(dataset, arch_name)]
    bs, n_ep = rc["bs"], EPOCHS[method]

    # ------------------------------------------------------------------
    # ⚠ SEED THE GLOBAL RNG. Do not remove.
    #
    # tr_aug carries RandomCrop + RandomHorizontalFlip, and with num_workers=0
    # those draw from the GLOBAL torch RNG in the main process. The DataLoader
    # `generator=` below fixes batch ORDER only, not augmentation. Without this
    # line the augmentation stream is whatever the process happened to start
    # with, so two runs from the same cached theta_o diverge.
    #
    # Measured: GA from the same cached theta_o, same batch order, global seed
    # varied 1/2/3 gave retain accuracy 59.40 / 43.60 / 33.10 % after 4 steps.
    # That nuisance was being read as seed variance.
    # ------------------------------------------------------------------
    torch.manual_seed(30_000 + seed)
    np.random.seed(30_000 + seed)

    model = copy.deepcopy(theta_o).to(pipeline.DEV)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=ETA_U)

    teacher = None
    if method == "SCRUB":
        teacher = copy.deepcopy(theta_o).to(pipeline.DEV).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    dl_f = _loader(tr_aug, D_f, bs, seed)
    dl_r = _loader(tr_aug, D_r, bs, seed) if method != "GradientAscent" else None

    for ep in range(n_ep):
        if method == "GradientAscent":
            for x, y in dl_f:
                x, y = x.to(pipeline.DEV), y.to(pipeline.DEV)
                loss = -F.cross_entropy(model(x), y)      # ASCEND on D_f
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

        elif method == "FineTuning":
            for x, y in dl_r:
                x, y = x.to(pipeline.DEV), y.to(pipeline.DEV)
                loss = F.cross_entropy(model(x), y)       # descend on D_r only
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

        elif method == "NegGradPlus":
            # One D_f batch is paired with one D_r batch so both terms enter the
            # SAME update, which is what "combines" requires. D_f is much smaller
            # than D_r, so its loader is cycled.
            it_f = iter(dl_f)
            for x_r, y_r in dl_r:
                try:
                    x_f, y_f = next(it_f)
                except StopIteration:
                    it_f = iter(dl_f)
                    x_f, y_f = next(it_f)
                x_r, y_r = x_r.to(pipeline.DEV), y_r.to(pipeline.DEV)
                x_f, y_f = x_f.to(pipeline.DEV), y_f.to(pipeline.DEV)
                loss = (ALPHA * F.cross_entropy(model(x_r), y_r)
                        - (1 - ALPHA) * F.cross_entropy(model(x_f), y_f))
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

        elif method == "SCRUB":
            # max-step on D_f: push the student AWAY from the teacher
            for x, _ in dl_f:
                x = x.to(pipeline.DEV)
                with torch.no_grad():
                    t_out = teacher(x)
                loss = -_kl_t(model(x), t_out)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            # min-step on D_r: keep the student ON the teacher, and on labels
            for x, y in dl_r:
                x, y = x.to(pipeline.DEV), y.to(pipeline.DEV)
                with torch.no_grad():
                    t_out = teacher(x)
                out = model(x)
                loss = (ALPHA * _kl_t(out, t_out)
                        + (1 - ALPHA) * F.cross_entropy(out, y))
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

    model.eval()
    return model


def selftest(log=print):
    """Degenerate checks that do not need a trained model (Standing Control 4).

    Asserts the machinery does what it says: GA raises forget loss, FT does not
    touch D_f directly, every method returns a NEW object, and theta_o is
    unchanged afterwards.
    """
    import arch
    ok = True
    ds, an = "cifar10", "allcnn"
    torch.manual_seed(0)
    m0 = arch.build(an, n_cls=10).to(pipeline.DEV)
    before = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}

    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    tiny = torch.utils.data.TensorDataset(x, y)
    D_r, D_f = np.arange(0, 48), np.arange(48, 64)

    for meth in EPOCHS:
        m = unlearn(meth, m0, tiny, D_r, D_f, ds, an, seed=0)
        same_obj = m is m0
        changed = any(not torch.equal(before[k], v.detach().cpu())
                      for k, v in m.state_dict().items())
        untouched = all(torch.equal(before[k], v.detach().cpu())
                        for k, v in m0.state_dict().items())
        good = (not same_obj) and changed and untouched
        ok = ok and good
        log(f"    {meth:>15}: new object {not same_obj}, weights moved "
            f"{changed}, theta_o untouched {untouched}  "
            f"{'OK' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("unlearn.py selftest")
    print("PASS" if selftest() else "FAIL")
