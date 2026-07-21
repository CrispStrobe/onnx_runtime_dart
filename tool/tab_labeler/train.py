# Train the symbolic tab-fingering labeler and export it to ONNX.
#
# A small CNN over a WINDOW of note-columns (each a multi-hot pitch-presence
# vector) predicts, per string, the human-chosen fret class — the SAME [6,21]
# per-string LogSoftmax contract TabCNN emits, so CometBeat's shipped
# tab_emission_decoder / arrangeTab DP consume it unchanged. Only the input is
# symbolic instead of audio.
#
# Op set is deliberately TabCNN's (Conv2d / ReLU / MaxPool / Flatten / Gemm /
# LogSoftmax) — already parity-verified on pure-Dart onnx_runtime_dart.
#
#   python train.py labeler_data.npz tab-labeler.onnx
import sys

import numpy as np
import torch
import torch.nn as nn

DATA = sys.argv[1] if len(sys.argv) > 1 else "labeler_data.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tab-labeler.onnx"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40

NUM_STRINGS, NUM_CLASSES = 6, 21


class Labeler(nn.Module):
    """Input [N, BINS, W, 1] (repo-native, like TabCNN) → [N, 6, 21] logits."""

    def __init__(self, bins, win):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, bins, win)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, NUM_STRINGS * NUM_CLASSES),
        )

    def forward(self, x):  # x: [N, BINS, W, 1]
        x = x.permute(0, 3, 1, 2)  # → [N, 1, BINS, W]
        x = self.conv(x)
        x = self.head(x)
        return x.view(-1, NUM_STRINGS, NUM_CLASSES)


class Export(nn.Module):
    """Wraps the backbone with the per-string LogSoftmax head (the DP's ABI)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.log_softmax(self.net(x), dim=-1)


def per_string_ce(logits, y):  # logits [N,6,21], y [N,6]
    return sum(
        nn.functional.cross_entropy(logits[:, s, :], y[:, s])
        for s in range(NUM_STRINGS)
    )


def evaluate(net, X, Y, bs=4096):
    net.eval()
    # Two metrics: per-string class accuracy (incl. silent), and note-level
    # string+fret accuracy over ACTUALLY-played strings (the meaningful one).
    tot_cell = cor_cell = tot_note = cor_note = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg = net(X[i:i + bs])
            pred = lg.argmax(-1)
            yb = Y[i:i + bs]
            cor_cell += (pred == yb).sum().item()
            tot_cell += yb.numel()
            played = yb > 0
            cor_note += ((pred == yb) & played).sum().item()
            tot_note += played.sum().item()
    return cor_cell / tot_cell, cor_note / max(tot_note, 1)


def main():
    d = np.load(DATA)
    Xtr = torch.tensor(d["Xtr"]); Ytr = torch.tensor(d["Ytr"])
    Xva = torch.tensor(d["Xva"]); Yva = torch.tensor(d["Yva"])
    bins, win = Xtr.shape[1], Xtr.shape[2]
    print(f"train {tuple(Xtr.shape)} | val {tuple(Xva.shape)} | bins {bins} win {win}")

    net = Labeler(bins, win)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)

    bs = 512
    idx = np.arange(len(Xtr))
    rng = np.random.default_rng(0)
    best = 0.0
    for ep in range(EPOCHS):
        net.train()
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = Xtr[b]; yb = Ytr[b]
            opt.zero_grad()
            loss = per_string_ce(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        cell, note = evaluate(net, Xva, Yva)
        best = max(best, note)
        if ep % 2 == 0 or ep == EPOCHS - 1:
            print(f"ep {ep:2d}  loss {tot / len(idx):.4f}  "
                  f"val cell-acc {cell:.4f}  val note(str+fret)-acc {note:.4f}")

    tr_cell, tr_note = evaluate(net, Xtr, Ytr)
    va_cell, va_note = evaluate(net, Xva, Yva)
    print(f"FINAL  train note-acc {tr_note:.4f}  val note-acc {va_note:.4f}  "
          f"(best val {best:.4f})")

    exp = Export(net).eval()
    dummy = torch.zeros(1, bins, win, 1)
    torch.onnx.export(
        exp, dummy, OUT,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13, dynamo=False,
    )
    print(f"exported {OUT}")
    # Stash a small parity fixture (val slice) for the Dart-side check.
    k = min(240, len(Xva))
    np.savez(OUT.replace(".onnx", "_parity.npz"),
             X=d["Xva"][:k].astype(np.float32),
             logprobs=torch.log_softmax(net(Xva[:k]), -1).detach().numpy())
    print(f"wrote parity fixture ({k} examples)")


if __name__ == "__main__":
    main()
