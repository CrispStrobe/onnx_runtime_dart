# Leave-one-guitarist-out CV for the symbolic labeler: train 6 folds (each of
# GuitarSet's 6 players held out once), report the per-fold best val string+fret
# accuracy + mean/std. Player 05 alone (the default val) can under- or over-state
# generalization; the spread tells us how noisy the single split is and where the
# true number sits. GuitarSet-only + ±AUG transposition aug (CC BY 4.0).
#
#   python logo_cv.py <anno-dir> [epochs] [aug]
import glob
import os
import sys

import jams
import numpy as np
import torch
import torch.nn as nn

ANNO = sys.argv[1] if len(sys.argv) > 1 else "anno"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 18
AUG = int(sys.argv[3]) if len(sys.argv) > 3 else 4

OPEN = [64, 59, 55, 50, 45, 40]
MAXFRET, WINDOW, HALF = 19, 9, 4
PLO, PHI = 40, 88
PBINS = PHI - PLO + 1
TOL = 0.045
NS, NC = 6, 21
torch.set_num_threads(min(8, os.cpu_count() or 4))
DEV = torch.device("cpu")


def columns_for(jam):
    ev = []
    for gs, anno in enumerate(jam.search(namespace="note_midi")):
        ds = 5 - gs
        for o in anno.data:
            m = int(round(o.value)); fr = m - OPEN[ds]
            if 0 <= fr <= MAXFRET:
                ev.append((float(o.time), ds, fr, m))
    ev.sort()
    cols, i = [], 0
    while i < len(ev):
        t0 = ev[i][0]; fr = {}; pi = set(); j = i
        while j < len(ev) and ev[j][0] - t0 <= TOL:
            _, ds, f, m = ev[j]
            fr.setdefault(ds, f); pi.add(m); j += 1
        cols.append((fr, pi)); i = j
    return cols


def shifted(cols, k):
    out = []
    for fr, pi in cols:
        nf = {}
        for ds, f in fr.items():
            f2 = f + k
            if not (0 <= f2 <= MAXFRET):
                return None
            nf[ds] = f2
        out.append((nf, {m + k for m in pi}))
    return out


def enc(pi):
    v = np.zeros(PBINS, np.uint8)
    for m in pi:
        if PLO <= m <= PHI:
            v[m - PLO] = 1
    return v


def emit(cols, Xs, Ys):
    e = [enc(p) for _, p in cols]
    for c in range(len(cols)):
        w = np.zeros((PBINS, WINDOW), np.uint8)
        for j in range(WINDOW):
            k = c - HALF + j
            if 0 <= k < len(cols):
                w[:, j] = e[k]
        y = np.zeros(6, np.int64)
        for ds, f in cols[c][0].items():
            y[ds] = f + 1
        Xs.append(w[:, :, None]); Ys.append(y)


def build(files, holdout):
    Xtr, Ytr, Xva, Yva = [], [], [], []
    for f in files:
        cols = columns_for(jams.load(f))
        if os.path.basename(f)[:2] == holdout:
            emit(cols, Xva, Yva)
        else:
            emit(cols, Xtr, Ytr)
            for k in range(-AUG, AUG + 1):
                if k and (sh := shifted(cols, k)) is not None:
                    emit(sh, Xtr, Ytr)
    return (np.asarray(Xtr, np.uint8), np.asarray(Ytr, np.int64),
            np.asarray(Xva, np.uint8), np.asarray(Yva, np.int64))


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3),
                            nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2))
        with torch.no_grad():
            flat = s.c(torch.zeros(1, 1, PBINS, WINDOW)).flatten(1).shape[1]
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(flat, 192),
                            nn.ReLU(), nn.Dropout(0.4), nn.Linear(192, NS * NC))

    def forward(s, x):
        return s.h(s.c(x.permute(0, 3, 1, 2))).view(-1, NS, NC)


def note_acc(net, X, Y):
    net.eval(); cn = tn = 0
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            p = net(torch.from_numpy(X[i:i + 4096].astype(np.float32))).argmax(-1)
            y = torch.from_numpy(Y[i:i + 4096]); pl = y > 0
            cn += ((p == y) & pl).sum().item(); tn += pl.sum().item()
    return cn / max(tn, 1)


def train_fold(Xtr, Ytr, Xva, Yva):
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, 15, 0.5)
    idx = np.arange(len(Xtr)); rng = np.random.default_rng(0); best = 0.0
    for ep in range(EPOCHS):
        net.train(); rng.shuffle(idx)
        for i in range(0, len(idx), 1024):
            b = idx[i:i + 1024]
            xb = torch.from_numpy(Xtr[b].astype(np.float32))
            yb = torch.from_numpy(Ytr[b])
            opt.zero_grad()
            lg = net(xb)
            loss = sum(nn.functional.cross_entropy(lg[:, s, :], yb[:, s]) for s in range(NS))
            loss.backward(); opt.step()
        sch.step()
        best = max(best, note_acc(net, Xva, Yva))
    return best


def main():
    torch.manual_seed(0); np.random.seed(0)
    files = sorted(glob.glob(os.path.join(ANNO, "*.jams")))
    accs = {}
    for h in ["00", "01", "02", "03", "04", "05"]:
        Xtr, Ytr, Xva, Yva = build(files, h)
        a = train_fold(Xtr, Ytr, Xva, Yva)
        accs[h] = a
        print(f"holdout {h}: val note-acc {a:.4f}  (train {len(Xtr)}, val {len(Xva)})",
              flush=True)
    v = np.array(list(accs.values()))
    print(f"\nLOGO-CV note-acc: mean {v.mean():.4f}  std {v.std():.4f}  "
          f"min {v.min():.4f}  max {v.max():.4f}")
    print(f"player 05 alone: {accs['05']:.4f}  (was our single-split val)")


if __name__ == "__main__":
    main()
