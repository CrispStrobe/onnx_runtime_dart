#!/usr/bin/env python3
"""Generates ORT-oracle parity fixtures for the Dart runtime tests.

Each fixture is a directory test/fixtures/<name>/ containing:
  model.onnx     — a tiny single-op (or few-op) ONNX model
  case.json      — {"inputs": {name: tensor}, "expected": {name: tensor}}
                   where tensor = {"dtype": "float32"|"int64", "shape": [...],
                                   "data": [flat values]}

Expected outputs are computed by native onnxruntime, so the Dart test suite
asserts byte-level parity against the reference implementation without needing
Python at test time. Regenerate with:  .venv/bin/python tool/gen_fixtures.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures"
OPSET = 17
RNG = np.random.default_rng(42)


def tensor_json(a: np.ndarray) -> dict:
    if a.dtype in (np.float32, np.float64):
        return {"dtype": "float32", "shape": list(a.shape),
                "data": [float(v) for v in a.astype(np.float32).ravel()]}
    return {"dtype": "int64", "shape": list(a.shape),
            "data": [int(v) for v in a.astype(np.int64).ravel()]}


def dtype_of(a: np.ndarray) -> int:
    if a.dtype == np.float32:
        return TensorProto.FLOAT
    if a.dtype == np.int32:
        return TensorProto.INT32
    if a.dtype == np.uint8:
        return TensorProto.UINT8
    if a.dtype == np.int8:
        return TensorProto.INT8
    return TensorProto.INT64


def emit(name: str, nodes, inputs: dict, initializers: dict | None = None,
         n_outputs: int = 1, opset: int = OPSET, ms_domain: bool = False):
    """Builds the model, runs ORT, writes the fixture directory."""
    initializers = initializers or {}
    out_names = [f"out{i}" for i in range(n_outputs)]
    if len(nodes) == 1 and len(nodes[0].output) == n_outputs:
        pass  # single node already wired to out0..outN
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(k, dtype_of(v), v.shape)
         for k, v in inputs.items()],
        [helper.make_empty_tensor_value_info(o) for o in out_names],
        initializer=[
            helper.make_tensor(k, dtype_of(v), v.shape, v.tobytes(), raw=True)
            for k, v in initializers.items()],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)] +
        ([helper.make_opsetid("com.microsoft", 1)] if ms_domain else []))
    model.ir_version = 8

    # ORT tolerates unshaped outputs; run first, then backfill the output
    # value infos with the observed dtype/shape so the strict checker passes.
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    expected = dict(zip(out_names, sess.run(out_names, {
        k: v for k, v in inputs.items()})))
    del model.graph.output[:]
    model.graph.output.extend([
        helper.make_tensor_value_info(o, dtype_of(expected[o]),
                                      expected[o].shape)
        for o in out_names])
    onnx.checker.check_model(model)

    d = FIXTURES / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(model.SerializeToString())
    (d / "case.json").write_text(json.dumps({
        "inputs": {k: tensor_json(v) for k, v in inputs.items()},
        "expected": {k: tensor_json(v) for k, v in expected.items()},
    }))
    print(f"  {name}: outputs {[list(v.shape) for v in expected.values()]}")


def f32(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def main():
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)

    # ---- existing-op regression coverage (sanity for the harness itself) ----
    emit("matmul_batched",
         [helper.make_node("MatMul", ["a", "b"], ["out0"])],
         {"a": f32(2, 3, 5, 7), "b": f32(2, 3, 7, 4)})
    emit("gemm_transB_bias",
         [helper.make_node("Gemm", ["x", "w", "c"], ["out0"],
                           transB=1, alpha=1.0, beta=1.0)],
         {"x": f32(4, 8)}, initializers={"w": f32(6, 8), "c": f32(6)})
    emit("add_bias_broadcast_lastdim",
         [helper.make_node("Add", ["a", "b"], ["out0"])],
         {"a": f32(2, 5, 8), "b": f32(8)})
    emit("add_broadcast_middle",
         [helper.make_node("Add", ["a", "b"], ["out0"])],
         {"a": f32(2, 1, 4, 3), "b": f32(5, 1, 3)})
    emit("softmax_lastaxis",
         [helper.make_node("Softmax", ["x"], ["out0"], axis=-1)],
         {"x": f32(2, 3, 7)})

    # ---- A1: Conv ----
    def conv(name, x, w, b=None, **attrs):
        ins = ["x", "w"] + (["b"] if b is not None else [])
        inits = {"w": w, **({"b": b} if b is not None else {})}
        emit(name, [helper.make_node("Conv", ins, ["out0"], **attrs)],
             {"x": x}, initializers=inits)

    conv("conv2d_basic", f32(1, 2, 7, 7), f32(3, 2, 3, 3))
    conv("conv2d_bias", f32(1, 2, 7, 7), f32(3, 2, 3, 3), b=f32(3))
    conv("conv2d_pads", f32(1, 2, 6, 6), f32(3, 2, 3, 3),
         pads=[1, 1, 1, 1])
    conv("conv2d_pads_asym", f32(1, 2, 6, 5), f32(3, 2, 3, 2),
         pads=[1, 0, 2, 1])
    conv("conv2d_strides", f32(1, 2, 9, 9), f32(3, 2, 3, 3),
         strides=[2, 2], pads=[1, 1, 1, 1])
    conv("conv2d_dilation", f32(1, 2, 9, 9), f32(3, 2, 3, 3),
         dilations=[2, 2])
    conv("conv2d_groups", f32(1, 4, 6, 6), f32(6, 2, 3, 3), group=2)
    conv("conv2d_depthwise", f32(1, 4, 6, 6), f32(4, 1, 3, 3), group=4,
         pads=[1, 1, 1, 1])
    conv("conv2d_same_upper", f32(1, 2, 7, 7), f32(3, 2, 3, 3),
         auto_pad="SAME_UPPER", strides=[2, 2])
    conv("conv2d_same_lower", f32(1, 2, 7, 7), f32(3, 2, 3, 3),
         auto_pad="SAME_LOWER", strides=[2, 2])
    conv("conv1d", f32(2, 3, 12), f32(4, 3, 5), pads=[2, 2])
    conv("conv3d", f32(1, 2, 5, 5, 5), f32(2, 2, 3, 3, 3),
         pads=[1, 1, 1, 1, 1, 1])

    # ---- A2: ConvTranspose ----
    def convt(name, x, w, b=None, **attrs):
        ins = ["x", "w"] + (["b"] if b is not None else [])
        inits = {"w": w, **({"b": b} if b is not None else {})}
        emit(name,
             [helper.make_node("ConvTranspose", ins, ["out0"], **attrs)],
             {"x": x}, initializers=inits)

    convt("convtranspose_basic", f32(1, 2, 5, 5), f32(2, 3, 3, 3))
    convt("convtranspose_stride2", f32(1, 2, 5, 5), f32(2, 3, 3, 3),
          strides=[2, 2])
    convt("convtranspose_pads", f32(1, 2, 5, 5), f32(2, 3, 3, 3),
          strides=[2, 2], pads=[1, 1, 1, 1])
    convt("convtranspose_outpad", f32(1, 2, 5, 5), f32(2, 3, 3, 3),
          strides=[2, 2], output_padding=[1, 1])
    convt("convtranspose_groups", f32(1, 4, 5, 5), f32(4, 2, 3, 3), group=2)
    convt("convtranspose_bias", f32(1, 2, 5, 5), f32(2, 3, 3, 3), b=f32(3))
    convt("convtranspose_1d", f32(2, 3, 8), f32(3, 2, 4), strides=[2])
    convt("convtranspose_output_shape", f32(1, 2, 5, 5), f32(2, 3, 3, 3),
          strides=[2, 2], output_shape=[1, 3, 10, 10])

    # ---- A1: pooling ----
    emit("maxpool_basic",
         [helper.make_node("MaxPool", ["x"], ["out0"], kernel_shape=[2, 2])],
         {"x": f32(1, 2, 6, 6)})
    emit("maxpool_stride_pads",
         [helper.make_node("MaxPool", ["x"], ["out0"], kernel_shape=[3, 3],
                           strides=[2, 2], pads=[1, 1, 1, 1])],
         {"x": f32(1, 2, 7, 7)})
    emit("maxpool_ceil",
         [helper.make_node("MaxPool", ["x"], ["out0"], kernel_shape=[3, 3],
                           strides=[2, 2], ceil_mode=1)],
         {"x": f32(1, 1, 8, 8)})
    emit("maxpool_dilation",
         [helper.make_node("MaxPool", ["x"], ["out0"], kernel_shape=[2, 2],
                           dilations=[2, 2])],
         {"x": f32(1, 2, 6, 6)})
    emit("maxpool_same_upper",
         [helper.make_node("MaxPool", ["x"], ["out0"], kernel_shape=[3, 3],
                           strides=[2, 2], auto_pad="SAME_UPPER")],
         {"x": f32(1, 2, 7, 7)})
    emit("avgpool_basic",
         [helper.make_node("AveragePool", ["x"], ["out0"],
                           kernel_shape=[3, 3])],
         {"x": f32(1, 2, 6, 6)})
    emit("avgpool_pads_exclude",
         [helper.make_node("AveragePool", ["x"], ["out0"], kernel_shape=[3, 3],
                           pads=[1, 1, 1, 1])],
         {"x": f32(1, 2, 5, 5)})
    emit("avgpool_pads_include",
         [helper.make_node("AveragePool", ["x"], ["out0"], kernel_shape=[3, 3],
                           pads=[1, 1, 1, 1], count_include_pad=1)],
         {"x": f32(1, 2, 5, 5)})
    emit("avgpool_ceil",
         [helper.make_node("AveragePool", ["x"], ["out0"], kernel_shape=[2, 2],
                           strides=[2, 2], ceil_mode=1)],
         {"x": f32(1, 1, 5, 5)})
    emit("global_avgpool",
         [helper.make_node("GlobalAveragePool", ["x"], ["out0"])],
         {"x": f32(2, 3, 5, 4)})
    emit("global_maxpool",
         [helper.make_node("GlobalMaxPool", ["x"], ["out0"])],
         {"x": f32(2, 3, 5, 4)})

    # ---- A1: normalization ----
    c = 4
    emit("batchnorm",
         [helper.make_node("BatchNormalization",
                           ["x", "s", "b", "m", "v"], ["out0"],
                           epsilon=1e-3)],
         {"x": f32(2, c, 5, 5)},
         initializers={"s": f32(c), "b": f32(c), "m": f32(c),
                       "v": np.abs(f32(c)) + 0.5})
    emit("instancenorm",
         [helper.make_node("InstanceNormalization", ["x", "s", "b"], ["out0"],
                           epsilon=1e-3)],
         {"x": f32(2, c, 5, 5)},
         initializers={"s": f32(c), "b": f32(c)})

    # ---- A1: Resize ----
    def resize(name, x, scales=None, sizes=None, **attrs):
        ins = ["x", "", "scales"] if scales is not None else \
              ["x", "", "", "sizes"]
        inits = {}
        if scales is not None:
            inits["scales"] = np.array(scales, dtype=np.float32)
        if sizes is not None:
            inits["sizes"] = np.array(sizes, dtype=np.int64)
        emit(name, [helper.make_node("Resize", ins, ["out0"], **attrs)],
             {"x": x}, initializers=inits)

    resize("resize_nearest_up", f32(1, 2, 4, 4), scales=[1, 1, 2, 2],
           mode="nearest")
    resize("resize_nearest_asym_floor", f32(1, 2, 4, 4), scales=[1, 1, 2, 2],
           mode="nearest", coordinate_transformation_mode="asymmetric",
           nearest_mode="floor")
    resize("resize_linear_half_pixel", f32(1, 2, 4, 4), scales=[1, 1, 2, 2],
           mode="linear")
    resize("resize_linear_align_corners", f32(1, 2, 4, 4),
           sizes=[1, 2, 7, 9], mode="linear",
           coordinate_transformation_mode="align_corners")
    resize("resize_linear_down", f32(1, 2, 8, 8), scales=[1, 1, 0.5, 0.5],
           mode="linear")

    # ---- A1: Flatten + activations ----
    emit("flatten_axis1",
         [helper.make_node("Flatten", ["x"], ["out0"])],
         {"x": f32(2, 3, 4, 5)})
    emit("flatten_axis0",
         [helper.make_node("Flatten", ["x"], ["out0"], axis=0)],
         {"x": f32(2, 3, 4)})
    emit("flatten_axis_neg",
         [helper.make_node("Flatten", ["x"], ["out0"], axis=-1)],
         {"x": f32(2, 3, 4)})
    emit("leakyrelu",
         [helper.make_node("LeakyRelu", ["x"], ["out0"], alpha=0.1)],
         {"x": f32(3, 4, 5)})
    emit("leakyrelu_default",
         [helper.make_node("LeakyRelu", ["x"], ["out0"])],
         {"x": f32(3, 4, 5)})
    emit("elu",
         [helper.make_node("Elu", ["x"], ["out0"], alpha=0.7)],
         {"x": f32(3, 4, 5)})
    emit("hardsigmoid",
         [helper.make_node("HardSigmoid", ["x"], ["out0"])],
         {"x": f32(3, 4, 5)})
    emit("hardswish",
         [helper.make_node("HardSwish", ["x"], ["out0"])],
         {"x": f32(3, 4, 5)})
    emit("argmax_axis1", [helper.make_node("ArgMax", ["x"], ["out0"], axis=1)],
         {"x": f32(2, 5, 3)}, opset=17)
    emit("argmin_nokeep_last",
         [helper.make_node("ArgMin", ["x"], ["out0"], axis=-1, keepdims=0,
                           select_last_index=1)],
         {"x": np.tile(f32(3, 1), (1, 4)).astype(np.float32)}, opset=17)
    emit("einsum_matmul", [helper.make_node("Einsum", ["a", "b"], ["out0"],
                                            equation="ij,jk->ik")],
         {"a": f32(3, 4), "b": f32(4, 5)}, opset=17)
    emit("einsum_batch_transpose",
         [helper.make_node("Einsum", ["a", "b"], ["out0"],
                           equation="bij,bkj->bik")],
         {"a": f32(2, 3, 4), "b": f32(2, 5, 4)}, opset=17)
    emit("einsum_outer_sum",
         [helper.make_node("Einsum", ["a", "b"], ["out0"],
                           equation="i,j->ij")],
         {"a": f32(4), "b": f32(5)}, opset=17)
    emit("einsum_trace_perm",
         [helper.make_node("Einsum", ["a"], ["out0"], equation="ijk->kji")],
         {"a": f32(2, 3, 4)}, opset=17)
    emit("einsum_reduce",
         [helper.make_node("Einsum", ["a"], ["out0"], equation="ijk->i")],
         {"a": f32(2, 3, 4)}, opset=17)
    emit("einsum_implicit",
         [helper.make_node("Einsum", ["a", "b"], ["out0"],
                           equation="ij,jk")],
         {"a": f32(3, 4), "b": f32(4, 2)}, opset=17)
    emit("groupnorm",
         [helper.make_node("GroupNormalization", ["x", "s", "b"], ["out0"],
                           num_groups=4, epsilon=1e-3)],
         {"x": f32(2, 8, 5, 5)},
         initializers={"s": f32(8), "b": f32(8)}, opset=21)
    grid = (RNG.uniform(-1.2, 1.2, (1, 6, 7, 2))).astype(np.float32)
    emit("gridsample_bilinear_zeros",
         [helper.make_node("GridSample", ["x", "g"], ["out0"],
                           mode="linear", padding_mode="zeros")],
         {"x": f32(1, 2, 5, 5)}, initializers={"g": grid}, opset=20)
    emit("gridsample_nearest_border_align",
         [helper.make_node("GridSample", ["x", "g"], ["out0"],
                           mode="nearest", padding_mode="border",
                           align_corners=1)],
         {"x": f32(1, 2, 5, 5)}, initializers={"g": grid}, opset=20)
    emit("gridsample_reflection",
         [helper.make_node("GridSample", ["x", "g"], ["out0"],
                           mode="linear", padding_mode="reflection")],
         {"x": f32(1, 2, 5, 5)}, initializers={"g": grid}, opset=20)
    rois = np.array([[0.5, 0.5, 3.5, 3.5], [1, 1, 6, 6]], dtype=np.float32)
    emit("roialign_avg",
         [helper.make_node("RoiAlign", ["x", "r", "bi"], ["out0"],
                           output_height=3, output_width=3,
                           spatial_scale=1.0, sampling_ratio=2)],
         {"x": f32(1, 2, 8, 8)},
         initializers={"r": rois, "bi": np.array([0, 0], dtype=np.int64)},
         opset=17)
    # sampling_ratio pinned to 1: ORT warns its own max-mode averaging is
    # wrong for other ratios (roialign.h "will be fixed in ORT 1.13"), so
    # only the ratio where the oracle is trustworthy is fixture-tested.
    emit("roialign_max_outhalf",
         [helper.make_node("RoiAlign", ["x", "r", "bi"], ["out0"],
                           output_height=2, output_width=2, mode="max",
                           sampling_ratio=1,
                           coordinate_transformation_mode="output_half_pixel",
                           spatial_scale=0.5)],
         {"x": f32(1, 2, 8, 8)},
         initializers={"r": rois * 2, "bi": np.array([0, 0], dtype=np.int64)},
         opset=17)
    emit("clip_attr_form",  # opset-6 Clip: min/max as attributes (Relu6)
         [helper.make_node("Clip", ["x"], ["out0"], min=0.0, max=6.0)],
         {"x": (f32(3, 7) * 5)}, opset=9)
    emit("transpose_default_perm",
         [helper.make_node("Transpose", ["x"], ["out0"])],
         {"x": f32(2, 3, 4)})
    emit("nonzero",
         [helper.make_node("NonZero", ["x"], ["out0"])],
         {}, initializers={"x": (RNG.integers(0, 3, (3, 5)) - 1).astype(
             np.float32)})
    emit("topk_largest",
         [helper.make_node("TopK", ["x", "k"], ["out0", "out1"], axis=-1)],
         {"x": f32(2, 3, 9)},
         initializers={"k": np.array([4], dtype=np.int64)}, n_outputs=2)
    emit("topk_smallest_axis0",
         [helper.make_node("TopK", ["x", "k"], ["out0", "out1"], axis=0,
                           largest=0)],
         {"x": f32(7, 4)},
         initializers={"k": np.array([3], dtype=np.int64)}, n_outputs=2)
    boxes = np.array([[[0, 0, 1, 1], [0, 0.05, 1, 1.05], [0, 2, 1, 3],
                       [0, 2.05, 1, 3.05], [5, 5, 6, 6]]], dtype=np.float32)
    nms_scores = np.array([[[0.9, 0.8, 0.7, 0.95, 0.3],
                            [0.1, 0.85, 0.6, 0.2, 0.75]]], dtype=np.float32)
    emit("nms_corners",
         [helper.make_node("NonMaxSuppression",
                           ["b", "s", "mx", "iou", "sc"], ["out0"])],
         {},
         initializers={"b": boxes, "s": nms_scores,
                       "mx": np.array([3], dtype=np.int64),
                       "iou": np.array([0.5], dtype=np.float32),
                       "sc": np.array([0.2], dtype=np.float32)})
    emit("nms_center",
         [helper.make_node("NonMaxSuppression", ["b", "s", "mx", "iou"],
                           ["out0"], center_point_box=1)],
         {},
         initializers={"b": np.abs(f32(1, 6, 4)) + 0.5,
                       "s": np.abs(f32(1, 1, 6)),
                       "mx": np.array([4], dtype=np.int64),
                       "iou": np.array([0.4], dtype=np.float32)})
    emit("trilu_upper",
         [helper.make_node("Trilu", ["x"], ["out0"])],
         {"x": f32(2, 4, 5)})
    emit("trilu_lower_k",
         [helper.make_node("Trilu", ["x", "k"], ["out0"], upper=0)],
         {"x": f32(3, 4)},
         initializers={"k": np.array(-1, dtype=np.int64)})
    emit("scatternd",
         [helper.make_node("ScatterND", ["d", "i", "u"], ["out0"])],
         {"d": f32(4, 3)},
         initializers={"i": np.array([[1], [3]], dtype=np.int64),
                       "u": f32(2, 3)})
    emit("scatternd_deep",
         [helper.make_node("ScatterND", ["d", "i", "u"], ["out0"])],
         {"d": f32(2, 3, 4)},
         initializers={"i": np.array([[0, 2], [1, 0]], dtype=np.int64),
                       "u": f32(2, 4)})
    emit("div_int64_truncates",
         [helper.make_node("Div", ["a", "b"], ["out0"])],
         {},
         initializers={"a": np.array([103, -103, 7, -7, 99, 100],
                                     dtype=np.int64),
                       "b": np.array([4, 4, 2, 2, 100, 100],
                                     dtype=np.int64)})
    emit("cast_fp16_roundtrip",
         [helper.make_node("Cast", ["x"], ["h"], to=TensorProto.FLOAT16),
          helper.make_node("Cast", ["h"], ["out0"], to=TensorProto.FLOAT)],
         {"x": (f32(3, 7) * 100)})
    emit("tile",
         [helper.make_node("Tile", ["x", "rep"], ["out0"])],
         {"x": f32(2, 3, 4)},
         initializers={"rep": np.array([2, 1, 3], dtype=np.int64)})
    emit("floor_ceil_round",
         [helper.make_node("Floor", ["x"], ["f"]),
          helper.make_node("Ceil", ["x"], ["c"]),
          helper.make_node("Round", ["x"], ["r"]),
          helper.make_node("Concat", ["f", "c", "r"], ["out0"], axis=0)],
         {"x": (f32(2, 6) * 3)})
    emit("softplus",
         [helper.make_node("Softplus", ["x"], ["out0"])],
         {"x": f32(3, 4, 5)})
    emit("prelu_channel_slope",
         [helper.make_node("PRelu", ["x", "s"], ["out0"])],
         {"x": f32(2, 3, 4, 5)}, initializers={"s": f32(3, 1, 1)})
    emit("gelu_erf",
         [helper.make_node("Gelu", ["x"], ["out0"])],
         {"x": f32(3, 4, 5)}, opset=20)
    emit("gelu_tanh",
         [helper.make_node("Gelu", ["x"], ["out0"], approximate="tanh")],
         {"x": f32(3, 4, 5)}, opset=20)

    # ---- A3: recurrent ops ----
    # Dim shorthand: seq=3, batch=2, input=4, hidden=5.
    S, B, I, H = 3, 2, 4, 5

    def rnn_like(name, op, gates, n_out, dirs=1, direction=None, bias=False,
                 seq_lens=None, init_h=False, init_c=False, peep=False,
                 **attrs):
        ins = ["x", "w", "r"]
        inits = {"w": f32(dirs, gates * H, I), "r": f32(dirs, gates * H, H)}
        opt = [("b", bias, lambda: f32(dirs, 2 * gates * H)),
               ("lens", seq_lens is not None,
                lambda: np.array(seq_lens, dtype=np.int32)),
               ("h0", init_h, lambda: f32(dirs, B, H)),
               ("c0", init_c, lambda: f32(dirs, B, H)),
               ("p", peep, lambda: f32(dirs, 3 * H))]
        for nm, use, make in opt:
            if use:
                ins.append(nm)
                inits[nm] = make()
            else:
                ins.append("")
        while ins and ins[-1] == "":
            ins.pop()
        if op != "LSTM":  # no c0/p inputs
            ins = ins[:6]
        if direction:
            attrs["direction"] = direction
        emit(name,
             [helper.make_node(op, ins, [f"out{i}" for i in range(n_out)],
                               hidden_size=H, **attrs)],
             {"x": f32(S, B, I)}, initializers=inits, n_outputs=n_out)

    rnn_like("lstm_forward", "LSTM", 4, 3)
    rnn_like("lstm_bias", "LSTM", 4, 3, bias=True)
    rnn_like("lstm_reverse", "LSTM", 4, 3, direction="reverse", bias=True)
    rnn_like("lstm_bidir", "LSTM", 4, 3, dirs=2, direction="bidirectional",
             bias=True)
    rnn_like("lstm_initial_hc", "LSTM", 4, 3, init_h=True, init_c=True)
    rnn_like("lstm_seqlens", "LSTM", 4, 3, bias=True, seq_lens=[3, 2])
    rnn_like("lstm_seqlens_reverse", "LSTM", 4, 3, direction="reverse",
             seq_lens=[2, 3])
    rnn_like("lstm_bidir_seqlens", "LSTM", 4, 3, dirs=2,
             direction="bidirectional", seq_lens=[3, 1], init_h=True,
             init_c=True)
    rnn_like("lstm_peepholes", "LSTM", 4, 3, bias=True, peep=True)
    rnn_like("gru_forward", "GRU", 3, 2)
    rnn_like("gru_bias", "GRU", 3, 2, bias=True)
    rnn_like("gru_lbr", "GRU", 3, 2, bias=True, linear_before_reset=1)
    rnn_like("gru_bidir", "GRU", 3, 2, dirs=2, direction="bidirectional",
             bias=True, init_h=True)
    rnn_like("gru_seqlens", "GRU", 3, 2, bias=True, seq_lens=[2, 3])
    rnn_like("rnn_forward", "RNN", 1, 2)
    rnn_like("rnn_reverse_bias", "RNN", 1, 2, direction="reverse", bias=True)
    rnn_like("rnn_bidir", "RNN", 1, 2, dirs=2, direction="bidirectional",
             bias=True, init_h=True)

    # ---- Pad + Size (needed by Silero VAD's STFT front-end) ----
    def pad(name, x, pads, mode="constant", value=None, axes=None, opset=OPSET):
        ins = ["x", "pads"]
        inits = {"pads": np.array(pads, dtype=np.int64)}
        if value is not None:
            ins.append("cv")
            inits["cv"] = np.array(value, dtype=np.float32)
        if axes is not None:
            if value is None:
                ins.append("")
            ins.append("axes")
            inits["axes"] = np.array(axes, dtype=np.int64)
        emit(name, [helper.make_node("Pad", ins, ["out0"], mode=mode)],
             {"x": x}, initializers=inits, opset=opset)

    pad("pad_constant_default", f32(2, 3), [1, 0, 0, 2])
    pad("pad_constant_value", f32(2, 3), [1, 1, 1, 1], value=7.5)
    pad("pad_reflect", f32(1, 1, 8), [0, 0, 3, 0, 0, 3], mode="reflect")
    pad("pad_edge", f32(2, 4), [0, 2, 1, 0], mode="edge")
    pad("pad_axes", f32(2, 3, 4), [2, 1], axes=[-1], opset=18)
    emit("size_op", [helper.make_node("Size", ["x"], ["out0"])],
         {"x": f32(2, 3, 4)})

    emit("reducemax_axes",
         [helper.make_node("ReduceMax", ["x"], ["out0"], axes=[1],
                           keepdims=1)],
         {"x": f32(2, 5, 3)}, opset=17)
    emit("reducemax_all_nokeep",
         [helper.make_node("ReduceMax", ["x"], ["out0"], keepdims=0)],
         {"x": f32(2, 5, 3)}, opset=17)
    emit("slice_attr_form",  # opset-9 Slice: starts/ends/axes as attributes
         [helper.make_node("Slice", ["x"], ["out0"], starts=[1, 0],
                           ends=[3, 2], axes=[0, 2])],
         {"x": f32(4, 3, 5)}, opset=9)
    emit("reduceprod_axes",
         [helper.make_node("ReduceProd", ["x"], ["out0"], axes=[0],
                           keepdims=1)],
         {"x": (f32(3, 4) * 0.5 + 1.0)}, opset=17)
    emit("reducesumsquare",
         [helper.make_node("ReduceSumSquare", ["x"], ["out0"], axes=[-1],
                           keepdims=1)],
         {"x": f32(2, 3, 5)}, opset=17)
    emit("split_equal",
         [helper.make_node("Split", ["x"], ["out0", "out1", "out2"], axis=1)],
         {"x": f32(2, 9, 4)}, n_outputs=3)
    emit("split_sizes",
         [helper.make_node("Split", ["x", "sp"], ["out0", "out1"], axis=-1)],
         {"x": f32(3, 7)},
         initializers={"sp": np.array([2, 5], dtype=np.int64)}, n_outputs=2)
    emit("split_uneven_numoutputs",
         [helper.make_node("Split", ["x"], ["out0", "out1", "out2"], axis=0,
                           num_outputs=3)],
         {"x": f32(8, 3)}, opset=18, n_outputs=3)
    # STFT (opset 17): windowed, hop 64, frame 128 (non-centered).
    emit("stft_window",
         [helper.make_node("STFT", ["x", "step", "win"], ["out0"])],
         {"x": f32(1, 400)},
         initializers={"step": np.array(64, dtype=np.int64),
                       "win": np.hanning(128).astype(np.float32)}, opset=17)
    emit("stft_no_window",
         [helper.make_node("STFT", ["x", "step", "", "flen"], ["out0"])],
         {"x": f32(2, 300)},
         initializers={"step": np.array(100, dtype=np.int64),
                       "flen": np.array(90, dtype=np.int64)}, opset=17)
    emit("reducemin_lastaxis",
         [helper.make_node("ReduceMin", ["x"], ["out0"], axes=[-1],
                           keepdims=1)],
         {"x": f32(4, 6)}, opset=17)

    # ---- fusion patterns (we fuse these; ORT runs them literally) ----
    emit("gelu_pattern_erf",
         [helper.make_node("Div", ["x", "sqrt2"], ["d"]),
          helper.make_node("Erf", ["d"], ["e"]),
          helper.make_node("Add", ["e", "one"], ["a"]),
          helper.make_node("Mul", ["x", "a"], ["m"]),
          helper.make_node("Mul", ["m", "half"], ["out0"])],
         {"x": f32(2, 5, 16)},
         initializers={"sqrt2": np.array(np.sqrt(2), dtype=np.float32),
                       "one": np.array(1.0, dtype=np.float32),
                       "half": np.array(0.5, dtype=np.float32)})
    # RMSNorm chain (Qwen/Maia style): x*rsqrt(mean(x^2)+eps)*gamma
    emit("rmsnorm_pattern",
         [helper.make_node("Pow", ["x", "two"], ["p"]),
          helper.make_node("ReduceMean", ["p", "ax"], ["m"], keepdims=1),
          helper.make_node("Add", ["m", "eps"], ["a"]),
          helper.make_node("Sqrt", ["a"], ["sq"]),
          helper.make_node("Reciprocal", ["sq"], ["r"]),
          helper.make_node("Mul", ["x", "r"], ["n"]),
          helper.make_node("Mul", ["n", "gamma"], ["out0"])],
         {"x": f32(2, 5, 16)},
         initializers={"two": np.array(2.0, dtype=np.float32),
                       "ax": np.array([2], dtype=np.int64),
                       "eps": np.array(1e-6, dtype=np.float32),
                       "gamma": f32(16)}, opset=18)
    # BERT-style attention block: scores/sqrt(d) + mask -> softmax -> context
    emit("sdpa_pattern",
         [helper.make_node("MatMul", ["q", "kt"], ["s0"]),
          helper.make_node("Div", ["s0", "sqrt_d"], ["s1"]),
          helper.make_node("Add", ["s1", "mask"], ["s2"]),
          helper.make_node("Softmax", ["s2"], ["p"], axis=-1),
          helper.make_node("MatMul", ["p", "v"], ["out0"])],
         {"q": f32(2, 3, 4, 8), "kt": f32(2, 3, 8, 4), "v": f32(2, 3, 4, 8),
          "mask": (RNG.standard_normal((2, 1, 1, 4)) * 4).astype(np.float32)},
         initializers={"sqrt_d": np.array(np.sqrt(8.0), dtype=np.float32)})

    # ---- A5: QDQ quantization ----
    emit("quantizelinear_uint8",
         [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["out0"])],
         {"x": f32(2, 3, 4) * 5},
         initializers={"s": np.array(0.05, dtype=np.float32),
                       "z": np.array(128, dtype=np.uint8)})
    emit("quantizelinear_int8",
         [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["out0"])],
         {"x": f32(2, 3, 4) * 5},
         initializers={"s": np.array(0.05, dtype=np.float32),
                       "z": np.array(-10, dtype=np.int8)})
    emit("quantizelinear_peraxis",
         [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["out0"],
                           axis=1)],
         {"x": f32(2, 3, 4) * 5},
         initializers={"s": np.array([0.02, 0.05, 0.1], dtype=np.float32),
                       "z": np.array([0, 10, 128], dtype=np.uint8)})
    emit("dequantizelinear_uint8",
         [helper.make_node("DequantizeLinear", ["x", "s", "z"], ["out0"])],
         {},
         initializers={"x": RNG.integers(0, 256, (2, 3, 4)).astype(np.uint8),
                       "s": np.array(0.05, dtype=np.float32),
                       "z": np.array(128, dtype=np.uint8)})
    emit("dequantizelinear_int8_peraxis",
         [helper.make_node("DequantizeLinear", ["x", "s", "z"], ["out0"],
                           axis=0)],
         {},
         initializers={"x": RNG.integers(-128, 128, (3, 4)).astype(np.int8),
                       "s": np.array([0.02, 0.05, 0.1], dtype=np.float32),
                       "z": np.array([-5, 0, 20], dtype=np.int8)})
    emit("dynamicquantizelinear",
         [helper.make_node("DynamicQuantizeLinear", ["x"],
                           ["out0", "out1", "out2"])],
         {"x": f32(3, 4) * 3}, n_outputs=3)
    # QDQ conv: int8 weights dequantize at load (constant-folded), input
    # goes through a Q->DQ pair, conv runs in float.
    emit("qdq_conv",
         [helper.make_node("QuantizeLinear", ["x", "sx", "zx"], ["xq"]),
          helper.make_node("DequantizeLinear", ["xq", "sx", "zx"], ["xdq"]),
          helper.make_node("DequantizeLinear", ["wq", "sw", "zw"], ["wdq"],
                           axis=0),
          helper.make_node("Conv", ["xdq", "wdq"], ["out0"],
                           pads=[1, 1, 1, 1])],
         {"x": f32(1, 2, 5, 5)},
         initializers={
             "sx": np.array(0.04, dtype=np.float32),
             "zx": np.array(128, dtype=np.uint8),
             "wq": RNG.integers(-128, 128, (3, 2, 3, 3)).astype(np.int8),
             "sw": np.array([0.01, 0.02, 0.015], dtype=np.float32),
             "zw": np.array([0, 0, 0], dtype=np.int8),
         })

    # ---- QOperator quantized ops ----
    u8 = lambda *s: RNG.integers(0, 256, s).astype(np.uint8)
    i8 = lambda *s: RNG.integers(-128, 128, s).astype(np.int8)
    emit("matmulinteger_u8i8",
         [helper.make_node("MatMulInteger", ["a", "b", "az", "bz"], ["out0"])],
         {},
         initializers={"a": u8(4, 6), "b": i8(6, 5),
                       "az": np.array(128, dtype=np.uint8),
                       "bz": np.array(3, dtype=np.int8)})
    emit("matmulinteger_nozp",
         [helper.make_node("MatMulInteger", ["a", "b"], ["out0"])],
         {}, initializers={"a": u8(3, 7), "b": u8(7, 4)})
    emit("matmulinteger_batched",
         [helper.make_node("MatMulInteger", ["a", "b", "az", "bz"], ["out0"])],
         {},
         initializers={"a": u8(2, 4, 6), "b": i8(2, 6, 5),
                       "az": np.array(100, dtype=np.uint8),
                       "bz": np.array(0, dtype=np.int8)})
    emit("matmulinteger_percol_bzp",
         [helper.make_node("MatMulInteger", ["a", "b", "az", "bz"], ["out0"])],
         {},
         initializers={"a": u8(4, 6), "b": i8(6, 5),
                       "az": np.array(77, dtype=np.uint8),
                       "bz": i8(5)})
    emit("convinteger",
         [helper.make_node("ConvInteger", ["x", "w", "xz", "wz"], ["out0"],
                           pads=[1, 1, 1, 1])],
         {},
         initializers={"x": u8(1, 2, 5, 5), "w": i8(3, 2, 3, 3),
                       "xz": np.array(120, dtype=np.uint8),
                       "wz": np.array(2, dtype=np.int8)})
    emit("convinteger_1d",
         [helper.make_node("ConvInteger", ["x", "w", "xz", "wz"], ["out0"],
                           pads=[2, 2])],
         {},
         initializers={"x": u8(1, 3, 12), "w": i8(4, 3, 5),
                       "xz": np.array(100, dtype=np.uint8),
                       "wz": np.array(-1, dtype=np.int8)})
    emit("qlinearmatmul",
         [helper.make_node("QLinearMatMul",
                           ["a", "as_", "az", "b", "bs", "bz",
                            "ys", "yz"], ["out0"])],
         {},
         initializers={"a": u8(4, 6), "as_": np.array(0.02, np.float32),
                       "az": np.array(113, dtype=np.uint8),
                       "b": u8(6, 5), "bs": np.array(0.05, np.float32),
                       "bz": np.array(128, dtype=np.uint8),
                       "ys": np.array(0.1, np.float32),
                       "yz": np.array(100, dtype=np.uint8)})
    emit("qlinearconv_pertensor",
         [helper.make_node("QLinearConv",
                           ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"],
                           ["out0"], pads=[1, 1, 1, 1])],
         {},
         initializers={"x": u8(1, 2, 5, 5), "xs": np.array(0.02, np.float32),
                       "xz": np.array(128, dtype=np.uint8),
                       "w": i8(3, 2, 3, 3), "ws": np.array(0.01, np.float32),
                       "wz": np.array(0, dtype=np.int8),
                       "ys": np.array(0.05, np.float32),
                       "yz": np.array(110, dtype=np.uint8)})
    emit("qlinearconv_perchannel_bias",
         [helper.make_node("QLinearConv",
                           ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz",
                            "bias"], ["out0"], strides=[2, 2])],
         {},
         initializers={"x": u8(1, 2, 7, 7), "xs": np.array(0.03, np.float32),
                       "xz": np.array(90, dtype=np.uint8),
                       "w": i8(4, 2, 3, 3),
                       "ws": np.array([0.01, 0.02, 0.005, 0.03], np.float32),
                       "wz": np.array([0, 1, -2, 0], dtype=np.int8),
                       "ys": np.array(0.07, np.float32),
                       "yz": np.array(128, dtype=np.uint8),
                       "bias": RNG.integers(-500, 500, 4).astype(np.int32)})

    # ---- MatMulNBits (com.microsoft, 4-bit block-quantized weights) ----
    def pack_q4(wq):  # [N, K] values 0..15 -> [N, nblocks, block/2] packed
        N, K = wq.shape
        nb = (K + 31) // 32
        padded = np.zeros((N, nb * 32), dtype=np.uint8)
        padded[:, :K] = wq
        blocks = padded.reshape(N, nb, 32)
        return (blocks[:, :, 0::2] | (blocks[:, :, 1::2] << 4)).astype(
            np.uint8)

    def mmnb(name, M, K, N, zp=False):
        wq = RNG.integers(0, 16, (N, K)).astype(np.uint8)
        nb = (K + 31) // 32
        inits = {"b": pack_q4(wq),
                 "s": (RNG.standard_normal(N * nb) * 0.05 + 0.2).astype(
                     np.float32)}
        ins = ["a", "b", "s"]
        if zp:
            zpv = RNG.integers(0, 16, N * nb).astype(np.uint8)
            packed = np.zeros((N * nb + 1) // 2, dtype=np.uint8)
            packed[: len(zpv) // 2] = (zpv[0::2][: len(zpv) // 2] |
                                       (zpv[1::2] << 4))
            if len(zpv) % 2:
                packed[-1] = zpv[-1]
            inits["z"] = packed
            ins.append("z")
        emit(name,
             [helper.make_node("MatMulNBits", ins, ["out0"], K=K, N=N,
                               bits=4, block_size=32,
                               domain="com.microsoft")],
             {"a": f32(M, K)}, initializers=inits, ms_domain=True)

    mmnb("matmulnbits_basic", 4, 64, 8)
    mmnb("matmulnbits_zeropoints", 3, 64, 6, zp=True)
    mmnb("matmulnbits_partial_block", 2, 40, 5)

    # ---- A4: control flow ----
    def vi(name, dt, shape):
        return helper.make_tensor_value_info(name, dt, shape)

    def if_fixture(name, cond_val):
        then_g = helper.make_graph(
            [helper.make_node("Add", ["x", "k"], ["res"])], "then_branch",
            [], [vi("res", TensorProto.FLOAT, [2, 3])])
        else_g = helper.make_graph(
            [helper.make_node("Sub", ["x", "k"], ["res_e"])], "else_branch",
            [], [vi("res_e", TensorProto.FLOAT, [2, 3])])
        emit(name,
             [helper.make_node("Cast", ["c"], ["cb"], to=TensorProto.BOOL),
              helper.make_node("If", ["cb"], ["out0"], then_branch=then_g,
                               else_branch=else_g)],
             {"x": f32(2, 3)},
             initializers={"k": f32(2, 3),
                           "c": np.array(cond_val, dtype=np.int64)})

    if_fixture("if_then", 1)
    if_fixture("if_else", 0)

    # Fixed trip count: v = x, 3 iterations of v *= k; scan collects each v.
    loop_body = helper.make_graph(
        [helper.make_node("Identity", ["cond_in"], ["cond_out"]),
         helper.make_node("Mul", ["v_in", "k"], ["v_out"]),
         helper.make_node("Identity", ["v_out"], ["scan_out"])],
        "loop_body",
        [vi("iter_num", TensorProto.INT64, []),
         vi("cond_in", TensorProto.BOOL, []),
         vi("v_in", TensorProto.FLOAT, [2, 3])],
        [vi("cond_out", TensorProto.BOOL, []),
         vi("v_out", TensorProto.FLOAT, [2, 3]),
         vi("scan_out", TensorProto.FLOAT, [2, 3])])
    emit("loop_fixed_trip",
         [helper.make_node("Loop", ["M", "", "x"], ["out0", "out1"],
                           body=loop_body)],
         {"x": f32(2, 3)},
         initializers={"k": f32(2, 3),
                       "M": np.array(3, dtype=np.int64)},
         n_outputs=2)

    # Condition-terminated: run while iter < 2 (3 iterations: 0, 1, 2),
    # v += x each time; captures outer x inside the body.
    cond_body = helper.make_graph(
        [helper.make_node("Less", ["iter_num", "limit"], ["cond_out"]),
         helper.make_node("Add", ["v_in", "x"], ["v_out"])],
        "cond_body",
        [vi("iter_num", TensorProto.INT64, []),
         vi("cond_in", TensorProto.BOOL, []),
         vi("v_in", TensorProto.FLOAT, [2, 3])],
        [vi("cond_out", TensorProto.BOOL, []),
         vi("v_out", TensorProto.FLOAT, [2, 3])])
    emit("loop_cond_terminated",
         [helper.make_node("Cast", ["one"], ["cb"], to=TensorProto.BOOL),
          helper.make_node("Loop", ["", "cb", "x"], ["out0"],
                           body=cond_body)],
         {"x": f32(2, 3)},
         initializers={"limit": np.array(2, dtype=np.int64),
                       "one": np.array(1, dtype=np.int64)})

    # ---- Scan ----
    # Running sum: state s [2]; scan input xs [4, 2]; per step s += x,
    # scan output collects each intermediate sum.
    scan_body = helper.make_graph(
        [helper.make_node("Add", ["s_in", "x_slice"], ["s_out"]),
         helper.make_node("Identity", ["s_out"], ["scan_out"])],
        "scan_body",
        [vi("s_in", TensorProto.FLOAT, [2]),
         vi("x_slice", TensorProto.FLOAT, [2])],
        [vi("s_out", TensorProto.FLOAT, [2]),
         vi("scan_out", TensorProto.FLOAT, [2])])
    emit("scan_cumsum",
         [helper.make_node("Scan", ["s0", "xs"], ["out0", "out1"],
                           body=scan_body, num_scan_inputs=1)],
         {"xs": f32(4, 2)},
         initializers={"s0": np.zeros(2, dtype=np.float32)},
         n_outputs=2)
    # Two scan inputs, capture of an outer initializer inside the body.
    scan_body2 = helper.make_graph(
        [helper.make_node("Mul", ["x_sl", "y_sl"], ["xy"]),
         helper.make_node("Add", ["s_in", "xy"], ["s_mid"]),
         helper.make_node("Add", ["s_mid", "k"], ["s_out"])],
        "scan_body2",
        [vi("s_in", TensorProto.FLOAT, [3]),
         vi("x_sl", TensorProto.FLOAT, [3]),
         vi("y_sl", TensorProto.FLOAT, [3])],
        [vi("s_out", TensorProto.FLOAT, [3])])
    emit("scan_two_inputs",
         [helper.make_node("Scan", ["s0", "xs", "ys"], ["out0"],
                           body=scan_body2, num_scan_inputs=2)],
         {"xs": f32(3, 3), "ys": f32(3, 3)},
         initializers={"s0": f32(3), "k": f32(3)})

    # Non-default axes/directions: slice xs along axis 1, reversed; stack
    # the scan output along axis 1, reversed.
    scan_body3 = helper.make_graph(
        [helper.make_node("Add", ["s_in", "x_slice"], ["s_out"]),
         helper.make_node("Identity", ["s_out"], ["scan_out"])],
        "scan_body3",
        [vi("s_in", TensorProto.FLOAT, [2]),
         vi("x_slice", TensorProto.FLOAT, [2])],
        [vi("s_out", TensorProto.FLOAT, [2]),
         vi("scan_out", TensorProto.FLOAT, [2])])
    emit("scan_axes_directions",
         [helper.make_node("Scan", ["s0", "xs"], ["out0", "out1"],
                           body=scan_body3, num_scan_inputs=1,
                           scan_input_axes=[1], scan_input_directions=[1],
                           scan_output_axes=[1], scan_output_directions=[1])],
         {"xs": f32(2, 5)},
         initializers={"s0": np.zeros(2, dtype=np.float32)},
         n_outputs=2)

    print("fixtures written to", FIXTURES)


if __name__ == "__main__":
    main()
