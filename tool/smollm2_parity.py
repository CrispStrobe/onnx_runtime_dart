"""Generate live-parity cases for a KV-cache LLM decoder (SmolLM2-135M).

Runs a prefill step (empty past) and a decode step (past = prefill's present)
through native ORT and writes case JSON consumable by tool/live_parity.dart.
This exercises real GroupQueryAttention KV-cache round-trip: present_key/value
outputs feeding back as next-step past_key/value.

  .venv/bin/python tool/smollm2_parity.py MODEL.onnx OUTDIR
"""
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort


def tj(arr):
    a = np.asarray(arr)
    if a.dtype == np.int64 or a.dtype == np.int32:
        return {"dtype": "int64", "shape": list(a.shape),
                "data": a.astype(np.int64).ravel().tolist()}
    return {"dtype": "float32", "shape": list(a.shape),
            "data": a.astype(np.float32).ravel().tolist()}


def main():
    path, outdir = sys.argv[1], sys.argv[2]
    m = onnx.load(path)
    n_layers = sum(1 for i in m.graph.input
                   if i.name.startswith("past_key_values") and i.name.endswith(".key"))
    # KV head count / head size from the past shape [batch, kvh, past, hs].
    pk = next(i for i in m.graph.input if i.name == "past_key_values.0.key")
    kvh = pk.type.tensor_type.shape.dim[1].dim_value
    hs = pk.type.tensor_type.shape.dim[3].dim_value
    vocab = m.graph.output[0].type.tensor_type.shape.dim[2].dim_value
    print(f"layers={n_layers} kv_heads={kvh} head_size={hs} vocab={vocab}")

    so = ort.SessionOptions()
    # Some fp16 exports trip ORT's own SimplifiedLayerNorm fusion at init;
    # disabling graph optimization gives a clean unfused oracle (which also
    # matches this runtime's unfused execution more directly).
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]

    rng = np.random.default_rng(0)
    seq = 6
    ids = rng.integers(1, min(vocab, 30000), size=(1, seq)).astype(np.int64)

    def empty_past():
        return {f"past_key_values.{l}.{kind}": np.zeros((1, kvh, 0, hs), np.float32)
                for l in range(n_layers) for kind in ("key", "value")}

    # --- prefill: empty past, positions 0..seq-1 ---
    pre = {
        "input_ids": ids,
        "attention_mask": np.ones((1, seq), np.int64),
        "position_ids": np.arange(seq, dtype=np.int64)[None, :],
        **empty_past(),
    }
    pre_out = sess.run(out_names, pre)
    pre_map = dict(zip(out_names, pre_out))
    _write(outdir + "/smollm2_prefill.json", pre, pre_map, n_layers)

    # --- decode: past = prefill present, one new token at position seq ---
    nxt = rng.integers(1, min(vocab, 30000), size=(1, 1)).astype(np.int64)
    dec = {
        "input_ids": nxt,
        "attention_mask": np.ones((1, seq + 1), np.int64),
        "position_ids": np.array([[seq]], np.int64),
    }
    for l in range(n_layers):
        dec[f"past_key_values.{l}.key"] = pre_map[f"present.{l}.key"]
        dec[f"past_key_values.{l}.value"] = pre_map[f"present.{l}.value"]
    dec_out = sess.run(out_names, dec)
    dec_map = dict(zip(out_names, dec_out))
    _write(outdir + "/smollm2_decode.json", dec, dec_map, n_layers)


def _write(fn, inputs, out_map, n_layers):
    # Verify logits + the layer-0 present KV (KV-cache correctness), keeping the
    # case small (logits alone is 49k floats/token).
    expected = {"logits": tj(out_map["logits"]),
                "present.0.key": tj(out_map["present.0.key"]),
                "present.0.value": tj(out_map["present.0.value"])}
    case = {"inputs": {k: tj(v) for k, v in inputs.items()}, "expected": expected}
    with open(fn, "w") as f:
        json.dump(case, f)
    print("wrote", fn, "logits shape", out_map["logits"].shape)


if __name__ == "__main__":
    main()
