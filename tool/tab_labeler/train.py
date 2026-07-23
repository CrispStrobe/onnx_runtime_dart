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
#   python train.py labeler_data.npz tab-labeler.onnx [epochs]
#
# Set WANDB=1 to log the run to Weights & Biases (project WANDB_PROJECT, default
# "tab-labeler"; creds from ~/.netrc). No-op / no dependency if unset.
import os
import sys

import numpy as np
import torch
import torch.nn as nn

DATA = sys.argv[1] if len(sys.argv) > 1 else "labeler_data.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tab-labeler.onnx"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
BEST_PT = OUT.replace(".onnx", ".best.pt")
SEED = int(os.environ.get("TAB_SEED", "0"))


def _wandb_init(config):
    """Return a W&B run when WANDB=1 (and the pkg imports), else None."""
    if os.environ.get("WANDB") not in ("1", "true", "yes"):
        return None
    try:
        import wandb
    except ImportError:
        print("WANDB set but wandb not installed — skipping")
        return None
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "tab-labeler"),
        name=os.environ.get("WANDB_NAME"),
        config=config,
    )

NUM_STRINGS, NUM_CLASSES = 6, 21

# Device: CPU by default. This model is TINY (conv on 49×9) so per-batch GPU
# kernel-launch overhead makes MPS *slower* than CPU here (a launch-bound tiny-op
# case — see the ggml dev notes). The OOM that motivated GPU was really the
# float32 dataset; uint8 storage fixes that, so CPU is both fast and memory-safe.
# Set TAB_DEVICE=mps to force the GPU.
_dev = os.environ.get("TAB_DEVICE", "cpu")
DEVICE = torch.device("mps") if _dev == "mps" and torch.backends.mps.is_available() \
    else torch.device("cpu")
torch.set_num_threads(min(8, os.cpu_count() or 4))


def _batch(Xu8, idx, device):
    """A dense float32 batch from uint8 storage — cast + move to the GPU here so
    the full dataset stays uint8 (4× smaller) on the host."""
    return torch.from_numpy(Xu8[idx].astype(np.float32)).to(device)


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
            nn.Linear(flat, 192), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(192, NUM_STRINGS * NUM_CLASSES),
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
            xb = _batch(X, slice(i, i + bs), DEVICE)
            pred = net(xb).argmax(-1).cpu()
            yb = torch.from_numpy(Y[i:i + bs])
            cor_cell += (pred == yb).sum().item()
            tot_cell += yb.numel()
            played = yb > 0
            cor_note += ((pred == yb) & played).sum().item()
            tot_note += played.sum().item()
    return cor_cell / tot_cell, cor_note / max(tot_note, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d = np.load(DATA)
    # Keep the dataset as uint8 on the host; batches cast to float on the GPU.
    Xtr = np.ascontiguousarray(d["Xtr"], np.uint8)
    Ytr = np.ascontiguousarray(d["Ytr"], np.int64)
    Xva = np.ascontiguousarray(d["Xva"], np.uint8)
    Yva = np.ascontiguousarray(d["Yva"], np.int64)
    bins, win = Xtr.shape[1], Xtr.shape[2]
    print(f"device {DEVICE} | train {Xtr.shape} ({Xtr.nbytes // 2**20} MB uint8) "
          f"| val {Xva.shape} | bins {bins} win {win}")

    net = Labeler(bins, win).to(DEVICE)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params: {nparams:,}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)

    run = _wandb_init({
        "data": DATA, "epochs": EPOCHS, "params": nparams, "device": str(DEVICE),
        "seed": SEED,
        "train_examples": len(Xtr), "val_examples": len(Xva),
        "bins": bins, "window": win,
    })

    bs = 1024  # larger batch → fewer dispatches, faster on CPU/MPS alike
    idx = np.arange(len(Xtr))
    rng = np.random.default_rng(0)
    best = 0.0
    best_state = None  # capture the peak — val is noisy across a held-out player
    for ep in range(EPOCHS):
        net.train()
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = _batch(Xtr, b, DEVICE)
            yb = torch.from_numpy(Ytr[b]).to(DEVICE)
            opt.zero_grad()
            loss = per_string_ce(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        cell, note = evaluate(net, Xva, Yva)
        if note > best:
            best = note
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "epoch": ep,
                "val_note_acc": best,
                "bins": bins,
                "window": win,
                "params": nparams,
            }, BEST_PT)
        if run:
            run.log({"epoch": ep, "loss": tot / len(idx),
                     "val_cell_acc": cell, "val_note_acc": note,
                     "lr": sched.get_last_lr()[0]})
        if ep % 2 == 0 or ep == EPOCHS - 1:
            print(f"ep {ep:2d}  loss {tot / len(idx):.4f}  "
                  f"val cell-acc {cell:.4f}  val note(str+fret)-acc {note:.4f}",
                  flush=True)

    # Restore the best-val checkpoint before export — the exported ONNX is the
    # peak model, not a noisier final epoch.
    if best_state is not None:
        net.load_state_dict(best_state)
    _, tr_note = evaluate(net, Xtr, Ytr)
    _, va_note = evaluate(net, Xva, Yva)
    print(f"FINAL (best ckpt)  train note-acc {tr_note:.4f}  "
          f"val note-acc {va_note:.4f}  (best val {best:.4f})")
    if run:
        run.summary["train_note_acc"] = tr_note
        run.summary["final_val_note_acc"] = va_note
        run.summary["best_val_note_acc"] = best

    # Export from CPU (the legacy ONNX exporter + onnxruntime want CPU tensors).
    net_cpu = net.to("cpu").eval()
    exp = Export(net_cpu).eval()
    torch.onnx.export(
        exp, torch.zeros(1, bins, win, 1), OUT,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13, dynamo=False,
    )
    print(f"exported {OUT}")
    k = min(240, len(Xva))
    with torch.no_grad():
        ref = torch.log_softmax(
            net_cpu(torch.from_numpy(Xva[:k].astype(np.float32))), -1).numpy()
    np.savez(OUT.replace(".onnx", "_parity.npz"),
             X=Xva[:k].astype(np.float32), logprobs=ref)
    print(f"wrote parity fixture ({k} examples)")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
