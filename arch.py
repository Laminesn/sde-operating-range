"""
Architectures matched EXACTLY to Zhang et al., ICLR 2026, Appendix D, Table 6.

WHY THIS FILE EXISTS
  Handoff step 1: "Parse their Appendix D and match architectures exactly.
  Biggest single threat to the findings."  The day-1..3 findings were obtained
  on an AllCNN-*style* net of our own construction (64/64/128/128/192 with
  MaxPool). If the findings are architecture artefacts they die here.

HOW THE ARCHITECTURE WAS IDENTIFIED
  The paper never writes the layer spec. It gives Table 6 — the activation
  DIMENSIONALITY of six taps — which is enough to pin it uniquely:

      Table 6            h1      h2      h3     h4    hp    h
      C10-AllCNN       98304   49152   12288  12288   192   10
      C10-ResNet18     65536   32768   16384   8192   512   10

  §4.1.3 names the AllCNN taps: "the output of four convolutional layers,
  Conv2, Conv4, Conv6, Conv8". Solving for a 32x32x3 input:

      Conv2 out 32x32x96  = 98304  -> h1   (so block-1 width is 96)
      Conv4 out 16x16x192 = 49152  -> h2   (Conv3 strides to 16, width 192)
      Conv6 out  8x8x192  = 12288  -> h3   (Conv6 strides to 8)
      Conv8 out  8x8x192  = 12288  -> h4   (Conv7 3x3, Conv8 1x1)
      hp        192                        (AvgPool2d(8) over Conv8)
      h          10                        (Linear 192 -> 10)

  All six match. That is All-CNN-C (Springenberg et al. 2015) at 96/192 width
  with BatchNorm and a global-average-pool + linear head — i.e. the AllCNN
  used throughout the unlearning literature, not the original paper's
  1x1-conv-to-logits variant (which would make hp = 10, contradicting Table 6).

  ResNet-18 taps are the four residual blocks (§4.1.3), CIFAR stem (3x3, no
  maxpool):  32x32x64 / 16x16x128 / 8x8x256 / 4x4x512 -> 65536/32768/16384/8192.
  Also an exact match.

VERIFY, DO NOT TRUST
  check_dims() asserts every tap against Table 6 and is called by smoke.py.
  Standing rule of this project: build the falsification before the method.
"""

import torch
import torch.nn as nn

# Appendix D, Table 6 — verbatim. The falsification target for this module.
TABLE6 = {
    ("allcnn", 32, 10): dict(h1=98304, h2=49152, h3=12288, h4=12288, hp=192, h=10),
    ("allcnn", 32, 100): dict(h1=98304, h2=49152, h3=12288, h4=12288, hp=192, h=100),
    ("resnet18", 32, 10): dict(h1=65536, h2=32768, h3=16384, h4=8192, hp=512, h=10),
    ("resnet18", 32, 100): dict(h1=65536, h2=32768, h3=16384, h4=8192, hp=512, h=100),
    # Tiny-ImageNet is 64x64: 262144/131072/65536/32768/512/200 in Table 6.
    ("resnet18", 64, 200): dict(h1=262144, h2=131072, h3=65536, h4=32768, hp=512, h=200),
}

TAPS = ("h1", "h2", "h3", "h4", "hp", "h")


def _conv(cin, cout, k, stride=1):
    """Conv-BN-ReLU. 'same' padding for odd k, so spatial size is set by stride."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class AllCNN(nn.Module):
    """All-CNN-C, 96/192 width, BN, GAP head. Matches Table 6 row C10-AllCNN."""

    def __init__(self, n_cls=10, in_ch=3, img=32):
        super().__init__()
        w1, w2 = 96, 192
        self.conv1 = _conv(in_ch, w1, 3)
        self.conv2 = _conv(w1, w1, 3)               # -> h1
        self.conv3 = _conv(w1, w2, 3, stride=2)
        self.conv4 = _conv(w2, w2, 3)               # -> h2
        self.conv5 = _conv(w2, w2, 3)
        self.conv6 = _conv(w2, w2, 3, stride=2)     # -> h3
        self.conv7 = _conv(w2, w2, 3)
        self.conv8 = _conv(w2, w2, 1)               # -> h4
        self.pool = nn.AdaptiveAvgPool2d(1)         # == AvgPool2d(8) at 32x32
        self.head = nn.Linear(w2, n_cls)
        self.feat_dim = w2

    def forward(self, x, tap=None):
        """tap=None -> logits. tap in TAPS -> that activation, flattened."""
        x = self.conv1(x)
        h1 = self.conv2(x)
        if tap == "h1":
            return h1.flatten(1)
        x = self.conv3(h1)
        h2 = self.conv4(x)
        if tap == "h2":
            return h2.flatten(1)
        x = self.conv5(h2)
        h3 = self.conv6(x)
        if tap == "h3":
            return h3.flatten(1)
        x = self.conv7(h3)
        h4 = self.conv8(x)
        if tap == "h4":
            return h4.flatten(1)
        hp = self.pool(h4).flatten(1)
        if tap == "hp":
            return hp
        return self.head(hp)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        o = self.act(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return self.act(o + self.short(x))


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stem, no maxpool. Matches Table 6."""

    def __init__(self, n_cls=10, in_ch=3, img=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make(64, 64, 1)
        self.layer2 = self._make(64, 128, 2)
        self.layer3 = self._make(128, 256, 2)
        self.layer4 = self._make(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, n_cls)
        self.feat_dim = 512

    @staticmethod
    def _make(cin, cout, stride):
        return nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))

    def forward(self, x, tap=None):
        x = self.stem(x)
        h1 = self.layer1(x)
        if tap == "h1":
            return h1.flatten(1)
        h2 = self.layer2(h1)
        if tap == "h2":
            return h2.flatten(1)
        h3 = self.layer3(h2)
        if tap == "h3":
            return h3.flatten(1)
        h4 = self.layer4(h3)
        if tap == "h4":
            return h4.flatten(1)
        hp = self.pool(h4).flatten(1)
        if tap == "hp":
            return hp
        return self.head(hp)


ARCHS = {"allcnn": AllCNN, "resnet18": ResNet18}


def build(arch, n_cls=10, in_ch=3, img=32):
    return ARCHS[arch](n_cls=n_cls, in_ch=in_ch, img=img)


def check_dims(arch, n_cls=10, img=32, verbose=True):
    """Assert every tap against Appendix D Table 6. Returns True or raises."""
    key = (arch, img, n_cls)
    if key not in TABLE6:
        raise KeyError(f"no Table 6 entry for {key}")
    want = TABLE6[key]
    m = build(arch, n_cls=n_cls, img=img).eval()
    x = torch.zeros(2, 3, img, img)
    ok = True
    with torch.no_grad():
        for t in TAPS:
            got = m(x, tap=None if t == "h" else t).shape[1]
            hit = got == want[t]
            ok &= hit
            if verbose:
                print(f"    {arch:>9} {t:>3}: got {got:>7}  Table 6 {want[t]:>7}  "
                      f"{'OK' if hit else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"{arch} does not match Appendix D Table 6")
    return True


if __name__ == "__main__":
    print("Architecture check against Zhang et al. ICLR 2026, Appendix D Table 6")
    for a, c, i in [("allcnn", 10, 32), ("allcnn", 100, 32),
                    ("resnet18", 10, 32), ("resnet18", 100, 32),
                    ("resnet18", 200, 64)]:
        check_dims(a, n_cls=c, img=i)
        print()
