"""Greedy KV-cache generation reference for SmolLM2. Emits prompt + the token
ids ORT generates so the Dart runtime can be checked for an identical decode.

  .venv/bin/python tool/smollm2_generate.py MODEL.onnx OUT.json [n_new]
"""
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort


def main():
    path, outfn = sys.argv[1], sys.argv[2]
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    m = onnx.load(path)
    n_layers = sum(1 for i in m.graph.input
                   if i.name.startswith("past_key_values") and i.name.endswith(".key"))
    pk = next(i for i in m.graph.input if i.name == "past_key_values.0.key")
    kvh = pk.type.tensor_type.shape.dim[1].dim_value
    hs = pk.type.tensor_type.shape.dim[3].dim_value

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]

    prompt = [1, 338, 24, 573, 8137, 297, 253]  # arbitrary fixed token ids
    ids = np.array([prompt], np.int64)
    past = {f"past_key_values.{l}.{k}": np.zeros((1, kvh, 0, hs), np.float32)
            for l in range(n_layers) for k in ("key", "value")}
    total = 0
    generated = []
    cur = ids
    for step in range(n_new):
        seq = cur.shape[1]
        feed = {
            "input_ids": cur,
            "attention_mask": np.ones((1, total + seq), np.int64),
            "position_ids": np.arange(total, total + seq, dtype=np.int64)[None, :],
            **past,
        }
        out = dict(zip(out_names, sess.run(out_names, feed)))
        nxt = int(np.argmax(out["logits"][0, -1]))
        generated.append(nxt)
        total += seq
        for l in range(n_layers):
            past[f"past_key_values.{l}.key"] = out[f"present.{l}.key"]
            past[f"past_key_values.{l}.value"] = out[f"present.{l}.value"]
        cur = np.array([[nxt]], np.int64)

    print("prompt   ", prompt)
    print("generated", generated)
    with open(outfn, "w") as f:
        json.dump({"prompt": prompt, "n_new": n_new,
                   "n_layers": n_layers, "kv_heads": kvh, "head_size": hs,
                   "generated": generated}, f)


if __name__ == "__main__":
    main()
