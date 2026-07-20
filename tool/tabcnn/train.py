# Train modern TabCNN on the preprocessed GuitarSet features and export the
# trained tabcnn.onnx (per-string LogSoftmax head). Single held-out guitarist
# fold (fast, one model — the paper's 6-fold is the eval protocol, not needed to
# ship a usable model). Reports the held-out frame accuracy.
import glob
import os
import sys

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow import keras

from model import (CON_WIN, N_BINS, NUM_CLASSES, NUM_STRINGS, avg_acc,
                   build_backbone, build_export_model, per_string_loss)

FEAT = sys.argv[1]
OUT_ONNX = sys.argv[2]
VAL_GUITARIST = int(os.environ.get("VAL_GUITARIST", "5"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
HALF = (CON_WIN - 1) // 2


def load_clips():
    clips = {}
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        cid = os.path.basename(p)[:-4]
        d = np.load(p)
        # pad frames axis by HALF each side (zeros) for windowing
        rep = np.pad(d["repr"], [(HALF, HALF), (0, 0)], mode="constant")
        clips[cid] = (rep.astype(np.float32), d["labels"].astype(np.float32))
    return clips


class Seq(keras.utils.Sequence):
    def __init__(self, clips, ids, batch=128, shuffle=True):
        super().__init__()
        self.clips = clips
        self.index = []  # (cid, frame)
        for cid in ids:
            n = self.clips[cid][1].shape[0]
            self.index += [(cid, f) for f in range(n)]
        self.batch = batch
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.index) // self.batch

    def on_epoch_end(self):
        self.order = np.arange(len(self.index))
        if self.shuffle:
            np.random.shuffle(self.order)

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        X = np.empty((len(idx), N_BINS, CON_WIN, 1), np.float32)
        y = np.empty((len(idx), NUM_STRINGS, NUM_CLASSES), np.float32)
        for j, k in enumerate(idx):
            cid, f = self.index[k]
            rep, lab = self.clips[cid]
            win = rep[f:f + CON_WIN]  # [9,192]
            X[j] = np.expand_dims(np.swapaxes(win, 0, 1), -1)  # [192,9,1]
            y[j] = lab[f]
        return X, y


def main():
    clips = load_clips()
    ids = list(clips.keys())
    val = [c for c in ids if int(c.split("_")[0]) == VAL_GUITARIST]
    trn = [c for c in ids if int(c.split("_")[0]) != VAL_GUITARIST]
    print(f"clips: {len(ids)}  train {len(trn)}  val {len(val)}")

    bb = build_backbone()
    bb.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0),
               loss=per_string_loss, metrics=[avg_acc])
    # Resume from the last checkpoint if present (RESUME=1) so a killed run
    # doesn't lose epochs.
    ckpt_dir = os.environ.get("CKPT_DIR", "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, "last.weights.h5")
    init_epoch = 0
    if os.environ.get("RESUME") == "1" and os.path.exists(last):
        bb.load_weights(last)
        init_epoch = int(os.environ.get("INIT_EPOCH", "0"))
        print(f"resumed from {last} at epoch {init_epoch}")
    tr, va = Seq(clips, trn), Seq(clips, val, shuffle=False)
    print(f"train batches {len(tr)}  val batches {len(va)}")
    # Per-epoch snapshots: 'last' (always) + 'best' (by val_avg_acc).
    cbs = [
        keras.callbacks.ModelCheckpoint(last, save_weights_only=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(ckpt_dir, "best.weights.h5"),
            save_weights_only=True, monitor="val_avg_acc",
            mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(
            os.path.join(ckpt_dir, "history.csv"), append=True),
    ]
    bb.fit(tr, validation_data=va, epochs=EPOCHS, initial_epoch=init_epoch,
           callbacks=cbs, verbose=2)

    res = bb.evaluate(va, verbose=0)
    print(f"HELD-OUT (guitarist {VAL_GUITARIST}) frame avg_acc: {res[1]:.4f}")

    m = build_export_model(bb)
    spec = (tf.TensorSpec((None, N_BINS, CON_WIN, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(m, input_signature=spec, opset=13,
                               output_path=OUT_ONNX)
    print("EXPORTED", OUT_ONNX)


if __name__ == "__main__":
    main()
