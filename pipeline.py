"""
Data + training, matched to Zhang et al. ICLR 2026 Appendix D.

APPENDIX D, VERBATIM
  "we used mini-batch stochastic gradient descent (SGD) with a weight decay of
   5 x 10^-4. The batch size was set to 256 for SVHN and CIFAR-10, and 128 for
   CIFAR-100 and Tiny-ImageNet. The initial learning rate was 0.01 for SVHN and
   0.1 for all other datasets. All models were trained for the number of epochs
   listed in Table 5."

  Table 5 epochs: SV-AllCNN 20, C10-AllCNN 50, SV-ResNet18 20,
                  C10-ResNet18 100, C100-ResNet18 100, Tiny-ResNet18 100

WHAT THE PAPER DOES NOT STATE  (recorded, because this is the class of omission
this whole phase exists to surface -- cf. the unstated HSIC estimator on day 1)
  * momentum                 -> 0.9  (universal default for this recipe)
  * lr schedule              -> cosine annealing to 0
  * data augmentation        -> random crop 32 pad 4 + horizontal flip (CIFAR),
                                crop only for SVHN (digits are not flip-invariant)
  * reference sets S_IT/S_OOT for Alg. 2 -> see protocol.py

  These are FALSIFIABLE against Table 5: the recipe must reproduce
  C10-AllCNN train 99.41 / test 91.63 and C10-ResNet18 train 99.94 / test 93.48.
  train.py prints the comparison every run. If the accuracies do not land, the
  unstated details are wrong and every downstream number is suspect.

PROTOCOL (Sec. 4.1)
  D_f = random R fraction of D_tr, R in {5%, 10%, 20%};  D_r = D_tr \\ D_f
  Train the RETRAINED model h^re on D_r only. D_f is never seen.
  So D_r is ground-truth in-training and D_f is ground-truth out-of-training,
  under a model that is otherwise trained identically. That is the controlled
  experiment, and it is why no unlearning algorithm is needed to test the metric.
"""

import ssl
import certifi
import os

# macOS python.org builds ship without system CA certs; torchvision download
# fails with CERTIFICATE_VERIFY_FAILED without this.
ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
    cafile=certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from arch import build

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data")

# Appendix D + Table 5. (dataset, arch) -> recipe.
RECIPE = {
    ("cifar10", "allcnn"):   dict(epochs=50,  lr=0.1,  bs=256, n_cls=10),
    ("cifar10", "resnet18"): dict(epochs=100, lr=0.1,  bs=256, n_cls=10),
    ("svhn", "allcnn"):      dict(epochs=20,  lr=0.01, bs=256, n_cls=10),
    ("svhn", "resnet18"):    dict(epochs=20,  lr=0.01, bs=256, n_cls=10),
    ("cifar100", "resnet18"): dict(epochs=100, lr=0.1, bs=128, n_cls=100),
}

# Table 5, for the reproduction check. (train, eval, test) in %.
TABLE5 = {
    ("svhn", "allcnn"):       (100.00, 94.84, 94.90),
    ("cifar10", "allcnn"):    (99.41, 91.95, 91.63),
    ("svhn", "resnet18"):     (100.00, 94.64, 94.85),
    ("cifar10", "resnet18"):  (99.94, 93.48, 93.48),
    ("cifar100", "resnet18"): (99.92, 72.75, 73.47),
}

# Table 7, train/test acc at 10/20/40/80% of training. The key evidence that
# their "training sufficiency" study never reaches partial memorisation.
#
# ⚠ THESE ARE R = 10% MODELS. Sec. 4.1.4: "In this experiment, we fix the
# forgetting ratio R = 10%, the target set size |S| = 1000". Table 2 likewise.
# Anything run at R = 5% is comparing against a model trained on 5% less data.
#
# ⚠ THE `test` ROW WAS MISSING UNTIL 2026-08-05 and is the more useful half: the
# TRAIN-TEST GAP is the causal variable for every membership-style statistic
# here, and it is only computable with both rows. CIFAR-100 is the one setting
# that traverses a real range (6.0 -> 26.6 pp); CIFAR-10 sits at ~6 pp
# throughout at any architecture.
TABLE7 = {
    ("svhn", "allcnn"):       dict(train=(88.09, 95.73, 98.51, 99.92),
                                   test=(88.16, 92.97, 92.54, 94.26)),
    ("cifar10", "allcnn"):    dict(train=(92.77, 99.21, 100.00, 100.00),
                                   test=(90.49, 92.07, 94.76, 94.83)),
    ("svhn", "resnet18"):     dict(train=(77.52, 84.92, 89.22, 98.65),
                                   test=(71.42, 78.12, 80.06, 91.49)),
    ("cifar10", "resnet18"):  dict(train=(76.56, 87.20, 91.50, 99.85),
                                   test=(71.90, 80.86, 85.16, 93.33)),
    ("cifar100", "resnet18"): dict(train=(58.68, 72.56, 81.99, 99.88),
                                   test=(52.65, 56.08, 57.14, 73.31)),
    ("tiny", "resnet18"):     dict(train=(60.30, 84.10, 91.04, 91.11),
                                   test=(53.94, 57.92, 57.75, 57.69)),
}

CKPT_FRACS = (0.10, 0.20, 0.40, 0.80)      # Table 2 / Table 7 checkpoints


# ------------------------------------------------------------------ data

def get_data(dataset="cifar10", augment=True):
    """Returns (train_ds_augmented, train_ds_clean, test_ds).

    train_ds_clean has no augmentation and is what activations are extracted
    from -- SDE measures dependence between representations, and augmentation
    noise there would be measuring the transform, not the model.
    """
    from torchvision import datasets, transforms

    stats = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    }[dataset]
    norm = transforms.Normalize(*stats)
    clean = transforms.Compose([transforms.ToTensor(), norm])

    aug_ops = [transforms.RandomCrop(32, padding=4)]
    if dataset != "svhn":                       # digits are not flip-invariant
        aug_ops.append(transforms.RandomHorizontalFlip())
    aug = transforms.Compose(aug_ops + [transforms.ToTensor(), norm])
    if not augment:
        aug = clean

    root = DATA_ROOT
    if dataset == "cifar10":
        tr_a = datasets.CIFAR10(root, train=True, download=True, transform=aug)
        tr_c = datasets.CIFAR10(root, train=True, download=True, transform=clean)
        te = datasets.CIFAR10(root, train=False, download=True, transform=clean)
    elif dataset == "cifar100":
        tr_a = datasets.CIFAR100(root, train=True, download=True, transform=aug)
        tr_c = datasets.CIFAR100(root, train=True, download=True, transform=clean)
        te = datasets.CIFAR100(root, train=False, download=True, transform=clean)
    elif dataset == "svhn":
        tr_a = datasets.SVHN(root, split="train", download=True, transform=aug)
        tr_c = datasets.SVHN(root, split="train", download=True, transform=clean)
        te = datasets.SVHN(root, split="test", download=True, transform=clean)
    else:
        raise ValueError(dataset)
    return tr_a, tr_c, te


def forget_split(n_train, R, seed, scheme="legacy"):
    """D_f / D_r split (Sec. 4.1). R is the forgetting ratio |D_f|/|D_tr|.

    ------------------------------------------------------------------------
    ⚠ THE `legacy` SCHEME PRODUCES NESTED FORGET SETS. Found 2026-08-04.
    ------------------------------------------------------------------------
    It draws ONE permutation from `rng(1000 + seed)` and slices the first
    `R * n_train` entries. The permutation does not depend on R, so at a fixed
    seed:

        D_f(5%)  is exactly a prefix of  D_f(10%)  is exactly a prefix of  D_f(20%)

    Verified for every seed. The consequence is statistical, not a crash: the
    three ratios at one seed are NESTED SUB-CONDITIONS OF ONE DRAW, not three
    independent replicates. A 3 x 5 grid that looks like n = 15 has an
    effective n of 5. Every pooled statistic over ratios is pseudoreplicated,
    and RULER's own A.14 notes the minimum achievable Wilcoxon p at n = 5 is
    0.062 -- so a design believing it has n = 15 cannot reach the significance
    it thinks it can.

    `legacy` IS KEPT AND IS THE DEFAULT ON PURPOSE. Every existing artefact --
    the 15 Phase 3 models, their 525 checkpoint records, the Phase A results,
    the M2 gate models -- stores or depends on splits generated this way.
    Changing the default would silently invalidate all of it, or trip the
    `assert np.array_equal(blob["D_r"], D_r)` guards in run_phaseA.py and
    run_m2gate5.py. Reproducing anything from before 2026-08-04 needs `legacy`.

    `independent` derives the stream from BOTH seed and R via a SeedSequence,
    so the three ratios are independent draws. Use it for all new work. Overlap
    between D_f(5%) and D_f(10%) drops from full containment (2500 records) to
    the chance level (~250).
    """
    n_f = int(round(R * n_train))
    if scheme == "legacy":
        rng = np.random.default_rng(1000 + seed)
    elif scheme == "independent":
        # R enters the seed material, so a different ratio is a different draw.
        # round(R * 1e5) keeps it an exact integer for the ratios in use.
        rng = np.random.default_rng(
            np.random.SeedSequence([1000 + seed, int(round(R * 100000))]))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    perm = rng.permutation(n_train)
    return perm[n_f:].copy(), perm[:n_f].copy()      # (D_r, D_f)


def forget_split_selftest(n_train=50000, seeds=(0, 1, 2, 3, 4),
                          ratios=(0.05, 0.10, 0.20), log=print):
    """Assert `legacy` nests and `independent` does not. Cheap, no training."""
    ok = True
    for scheme, want_nested in (("legacy", True), ("independent", False)):
        nested = 0
        overlaps = []
        for s in seeds:
            fs = {R: set(forget_split(n_train, R, s, scheme)[1]) for R in ratios}
            rs = sorted(ratios)
            for a, b in zip(rs[:-1], rs[1:]):
                if fs[a] <= fs[b]:
                    nested += 1
                overlaps.append(len(fs[a] & fs[b]))
        n_pairs = len(seeds) * (len(ratios) - 1)
        got_nested = nested == n_pairs
        good = got_nested == want_nested
        ok = ok and good
        log(f"    {scheme:>12}: nested {nested}/{n_pairs} pairs, "
            f"mean overlap {np.mean(overlaps):.0f} records  "
            f"{'OK' if good else 'FAIL'}")
    # independence sanity: expected overlap of D_f(5%) and D_f(10%) is
    # 0.05 * 0.10 * n = 250 at n = 50000
    a = set(forget_split(n_train, 0.05, 0, "independent")[1])
    b = set(forget_split(n_train, 0.10, 0, "independent")[1])
    exp = 0.05 * 0.10 * n_train
    close = abs(len(a & b) - exp) < 4 * np.sqrt(exp)
    ok = ok and close
    log(f"    independent overlap {len(a & b)} vs chance {exp:.0f}  "
        f"{'OK' if close else 'FAIL'}")
    # both schemes must still partition correctly
    for scheme in ("legacy", "independent"):
        D_r, D_f = forget_split(n_train, 0.10, 0, scheme)
        part = (len(D_r) + len(D_f) == n_train
                and not set(D_r) & set(D_f)
                and len(D_f) == int(round(0.10 * n_train)))
        ok = ok and part
        log(f"    {scheme:>12}: partition valid {'OK' if part else 'FAIL'}")
    return ok


# ------------------------------------------------------------------ train

@torch.no_grad()
def accuracy(model, ds, idx=None, bs=512):
    model.eval()
    d = ds if idx is None else Subset(ds, idx)
    c = t = 0
    for x, y in DataLoader(d, batch_size=bs, num_workers=0):
        c += (model(x.to(DEV)).argmax(1).cpu() == y).sum().item()
        t += len(y)
    return c / max(t, 1)


@torch.no_grad()
def activations(model, ds, idx, tap="hp", bs=512):
    """h_tap(x) for the given indices, as float64 (n, dim)."""
    model.eval()
    out = []
    for x, _ in DataLoader(Subset(ds, idx), batch_size=bs, num_workers=0):
        z = model(x.to(DEV), tap=None if tap == "h" else tap)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def labels_of(ds):
    """Ground-truth labels as an int array, across torchvision's naming."""
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    raise AttributeError("dataset exposes no targets/labels")


def checkpoint_epochs(epochs, dense_first=10):
    """Which epochs to stop and evaluate at.

    Two groups, for two different experiments:
      * Table 2 / Table 7 fractions (10/20/40/80%) -- replicates their
        training-sufficiency study exactly.
      * EVERY epoch up to `dense_first`, plus the final epoch -- this is the
        continuous memorisation sweep (handoff step 4). It is dense early
        because that is the only place partial memorisation exists: Table 7
        shows C10-AllCNN is already at 92.77% train accuracy by their 10%
        checkpoint and 100.00% by 40%, so their sufficiency study never
        observes a partially-memorising model at all.

    ⚠ THIS DEFAULT WAS NOT USED FOR THE CANONICAL PHASE 3 SWEEP. The claim
    above that "the regime lives in epochs 1-10" is WRONG and was corrected on
    2026-08-03. The dense run passed --ckpt-list explicitly and used
        [1, 5, 10, 11..40, 45, 50]      (35 epochs, dense over 11-40)
    because the 0.90-0.99 band is crossed between epochs 20 and 40, not early.
    This function returns only 13 epochs [1..10, 20, 40, 50], seven of which
    (2,3,4,6,7,8,9) appear in NO Phase 3 record. Anything that must join to the
    canonical results has to pass the recorded list explicitly; see
    run_phase0.PHASE3_CKPTS.
    """
    eps = {max(1, int(round(f * epochs))) for f in CKPT_FRACS}
    eps |= set(range(1, min(dense_first, epochs) + 1))
    eps.add(epochs)
    return sorted(eps)


def train(dataset, arch, D_r, seed, epochs=None, log=print,
          tr_aug=None, tr_clean=None, te=None, eval_idx=None,
          on_checkpoint=None, dense_first=10, workers=0,
          ckpt_list=None, exact_acc=True, stop_after=None, order_seed=None):
    """Train the retrained model h^re on D_r with the Appendix D recipe.

    on_checkpoint(model, info) is called at each checkpoint epoch. Evaluating
    inline avoids holding or writing state dicts: at 5 seeds x 3 ratios x 2
    architectures the checkpoint tensors alone would be ~12 GB.

    stop_after: break after this epoch WITHOUT changing the LR schedule. The
    schedule is CosineAnnealingLR(T_max=epochs), so passing a smaller `epochs`
    to shorten a run silently changes every learning rate and produces a
    different trajectory. Any partial run that must match a full run's early
    epochs has to keep `epochs` at the recipe value and stop early instead.
    Used by trajectory_gate(); see RESEARCH-METHODOLOGY.md Revision 9.

    order_seed: controls BATCH ORDER independently of `seed`, which controls
    weight initialisation. Defaults to `seed`, so existing behaviour is
    unchanged. Added for RULER's A.5/A.6 control: their M2 assumes a paired-seed
    design (theta_o and theta_r share an initialisation), and A.6 reports that
    "the paired-seed calibration baseline is invalidated when batch ordering
    introduces stochastic variation between the original model and oracle".
    Testing that claim needs two models with the SAME init and the SAME data but
    DIFFERENT batch order, which is impossible while one seed controls both.

    Returns (model, history); history entries carry accuracies, no weights.
    """
    rc = RECIPE[(dataset, arch)]
    epochs = epochs or rc["epochs"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build(arch, n_cls=rc["n_cls"]).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=rc["lr"], momentum=0.9,
                          weight_decay=5e-4)                     # App. D
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # ---------------------------------------------------------------------
    # RNG ISOLATION -- do not remove. Without it, MEASUREMENT PERTURBS THE
    # MODEL BEING MEASURED.
    #
    # Two separate channels leak evaluation randomness into training, both via
    # the GLOBAL torch RNG:
    #   1. batch order  -- the training loader draws a base seed from the global
    #      RNG once per epoch, and so does every shuffle=False loader used by
    #      accuracy() and ActBank.
    #   2. augmentation -- with num_workers=0, RandomCrop and
    #      RandomHorizontalFlip draw from the global RNG in the main process.
    #
    # So the NUMBER OF CHECKPOINTS TAKEN changes how far the global RNG has
    # advanced, which changes every later epoch's batches and augmentations,
    # which changes the trained model. Observed directly: identical seed and
    # recipe, 13 checkpoints vs 35, gave train accuracy 0.8352 vs 0.7106 at
    # epoch 10. Same seed, different model.
    #
    # Fixed two ways, belt and braces:
    #   * the training loader gets its own generator, so batch order is a
    #     function of `seed` alone;
    #   * the global torch/numpy RNG state is snapshotted before each
    #     checkpoint and restored after, so nothing evaluation does can
    #     advance the stream that training consumes.
    # Verified by train_rng_isolated() below: any checkpoint schedule now
    # reproduces the same trajectory.
    # ---------------------------------------------------------------------
    gen = torch.Generator()
    gen.manual_seed(10_000 + (seed if order_seed is None else order_seed))

    # workers=0 by default: measured at 31.1 s/epoch either way (the step is
    # MPS-compute-bound, and augmentation on 6 cores is 0.8 s/epoch warm), so
    # extra workers only add ~15 s of one-time spawn cost per run.
    dl = DataLoader(Subset(tr_aug, D_r), batch_size=rc["bs"], shuffle=True,
                    num_workers=workers, drop_last=False,
                    persistent_workers=bool(workers), generator=gen)

    # `if ckpt_list` would treat an EMPTY list as "not supplied" and silently
    # substitute the 14-epoch default, so a run asking for no checkpoints would
    # quietly take fourteen. Found by the CIFAR-100 audit, 2026-08-05.
    ckpt_eps = set(ckpt_list) if ckpt_list is not None else set(
        checkpoint_epochs(epochs, dense_first))
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = cor = 0
        ls = 0.0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            o = model(x)
            loss = F.cross_entropy(o, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            ls += loss.item() * len(y)
            cor += (o.argmax(1) == y).sum().item()
            tot += len(y)
        sched.step()

        if ep in ckpt_eps:
            # snapshot the global RNG so measurement cannot advance the stream
            # training consumes (see RNG ISOLATION above)
            _t_state = torch.get_rng_state()
            _n_state = np.random.get_state()

            # Clean-view accuracies: the augmented running accuracy understates
            # memorisation, and memorisation is the independent variable here.
            # exact_acc=False leaves them to on_checkpoint, which derives them
            # from the activation bank for free (logits = head(hp)) instead of
            # paying a separate ~12 s forward pass over 60k images.
            if exact_acc:
                a_tr = accuracy(model, tr_clean, D_r)
                a_ev = accuracy(model, tr_clean, eval_idx) if eval_idx is not None else float("nan")
                a_te = accuracy(model, te)
            else:
                a_tr = a_ev = a_te = float("nan")
            info = dict(epoch=ep, frac=ep / epochs, train_acc=a_tr,
                        eval_acc=a_ev, test_acc=a_te,
                        running_acc=cor / tot, loss=ls / tot)
            if on_checkpoint is not None:
                info.update(on_checkpoint(model, info) or {})
            log(f"    ep {ep:>3}/{epochs}  loss={ls/tot:.4f}  "
                f"train(D_r)={info['train_acc']:.4f}  "
                f"D_f={info['eval_acc']:.4f}  test={info['test_acc']:.4f}")
            hist.append(info)

            torch.set_rng_state(_t_state)
            np.random.set_state(_n_state)
            model.train()

        if stop_after is not None and ep >= stop_after:
            break
    return model, hist


def train_rng_isolated(dataset="cifar10", arch="allcnn", R=0.05, seed=0,
                       epochs=10, data=None, log=print):
    """Assert the checkpoint schedule does not change the trained model.

    Trains twice with deliberately different schedules and compares train
    accuracy at the epochs they share. Must be exact. This guards the bug
    documented under RNG ISOLATION, which silently made two runs with the same
    seed produce different models.
    """
    tr_aug, tr_clean, te = data or get_data(dataset)
    D_r, D_f = forget_split(len(tr_clean), R, seed)
    saved = RECIPE[(dataset, arch)]
    RECIPE[(dataset, arch)] = {**saved, "epochs": epochs}
    try:
        got = {}
        for name, ck in (("sparse", [epochs // 2, epochs]),
                         ("dense", list(range(1, epochs + 1)))):
            _, h = train(dataset, arch, D_r, seed, tr_aug=tr_aug,
                         tr_clean=tr_clean, te=te, eval_idx=D_f,
                         ckpt_list=ck, exact_acc=True, log=lambda *a: None)
            got[name] = {x["epoch"]: x["train_acc"] for x in h}
            log(f"    {name:>7}: " + "  ".join(
                f"ep{e}={v:.4f}" for e, v in sorted(got[name].items())
                if e in (epochs // 2, epochs)))
    finally:
        RECIPE[(dataset, arch)] = saved
    a, b = got["sparse"], got["dense"]
    shared = sorted(set(a) & set(b))
    d = max(abs(a[e] - b[e]) for e in shared)
    log(f"    max |diff| at shared epochs {shared}: {d:.8f}")
    ok = d < 1e-9
    log("    -> RNG ISOLATED" if ok else "    -> LEAKING: schedule changes the model")
    return ok


def trajectory_gate(dataset="cifar10", arch="allcnn", R=0.05, seed=0,
                    check_epochs=(1, 5), tol=1e-6, results_dir=None,
                    data=None, log=print):
    """Standing Control 8: gate on the TRAJECTORY, not the endpoint.

    Retrains one recorded config and asserts the train accuracy at `check_epochs`
    reproduces the value stored in that config's Phase 3 results JSON.

    Why this exists. The SDE reproduction gate matched the paper's FINAL test
    accuracy to ~0.6pp and passed, while the training trajectory was completely
    different: their Table 7 has 92.77% train accuracy at epoch 5 where ours has
    64-76%, almost certainly because of unstated data augmentation. When the
    quantity under study is a property of training (here: memorisation level), a
    final-accuracy anchor is nearly invariant to the unstated parts of a recipe
    and cannot detect that. See RESEARCH-METHODOLOGY.md Revision 9.

    Used here for a second purpose: environment drift. Phase 4 reruns the Phase 3
    sweep to save checkpoints, and those checkpoints are only comparable to the
    Phase 3 records if the trajectory still reproduces bit-for-bit under the
    current torch/MPS build. Anything above `tol` means it does not, and nothing
    downstream may be compared across phases.

    `check_epochs` must be epochs that exist in the recorded schedule, which is
    [1, 5, 10, 11..40, 45, 50] -- dense over 11-40, NOT over 1-10.
    """
    import glob
    import json
    import os

    results_dir = results_dir or os.path.join(HERE, "results")
    tag = f"{dataset}-{arch}-R{int(round(R*100)):02d}-s{seed}-"
    hits = sorted(glob.glob(os.path.join(results_dir, tag + "*-dense.json")))
    if not hits:
        log(f"    no recorded run for {tag}* in {results_dir} -- gate SKIPPED")
        return None
    rec = json.load(open(hits[0]))
    ref = {h["epoch"]: h["train_acc"] for h in rec["history"]}

    missing = [e for e in check_epochs if e not in ref]
    if missing:
        raise ValueError(
            f"epochs {missing} are not in the recorded schedule "
            f"{sorted(ref)}; pick epochs that were actually checkpointed")

    tr_aug, tr_clean, te = data or get_data(dataset)
    D_r, D_f = forget_split(len(tr_clean), R, seed)

    # epochs stays at the recipe value so CosineAnnealingLR(T_max) is unchanged;
    # stop_after truncates the loop. Passing a smaller `epochs` here would alter
    # every learning rate and the comparison would be meaningless.
    _, h = train(dataset, arch, D_r, seed, tr_aug=tr_aug, tr_clean=tr_clean,
                 te=te, eval_idx=D_f, ckpt_list=list(check_epochs),
                 stop_after=max(check_epochs), exact_acc=True,
                 log=lambda *a: None)
    got = {x["epoch"]: x["train_acc"] for x in h}

    worst = 0.0
    for e in check_epochs:
        d = abs(got[e] - ref[e])
        worst = max(worst, d)
        log(f"    ep {e:>3}: recorded {ref[e]:.10f}  now {got[e]:.10f}  "
            f"|diff|={d:.2e}")
    ok = worst <= tol
    log(f"    max |diff| = {worst:.2e}  (tol {tol:.0e})")
    log("    -> TRAJECTORY REPRODUCES" if ok else
        "    -> TRAJECTORY DRIFTED: do not compare new runs to Phase 3 records")
    return ok


def compare_table5(dataset, arch, train_acc, eval_acc, test_acc, log=print):
    """Falsification of the unstated training details against Table 5."""
    key = (dataset, arch)
    if key not in TABLE5:
        return None
    t5 = TABLE5[key]
    got = (100 * train_acc, 100 * eval_acc, 100 * test_acc)
    log(f"  Table 5 check [{dataset}-{arch}]")
    log(f"    {'':>10}{'ours':>9}{'paper':>9}{'delta':>9}")
    for name, g, p in zip(("TRAIN", "EVAL", "TEST"), got, t5):
        log(f"    {name:>10}{g:>9.2f}{p:>9.2f}{g - p:>+9.2f}")
    # test accuracy is the discriminating one; train saturates for everyone
    ok = abs(got[2] - t5[2]) <= 2.0
    log(f"    -> test within 2pp of paper: {'OK' if ok else 'OFF - unstated details wrong'}")
    return ok
