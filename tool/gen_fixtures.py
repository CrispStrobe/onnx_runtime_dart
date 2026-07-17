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
    return TensorProto.FLOAT if a.dtype == np.float32 else TensorProto.INT64


def emit(name: str, nodes, inputs: dict, initializers: dict | None = None,
         n_outputs: int = 1, opset: int = OPSET):
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
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, None)
         for o in out_names],
        initializer=[
            helper.make_tensor(k, dtype_of(v), v.shape,
                               v.astype(np.float32).tobytes()
                               if v.dtype == np.float32 else
                               v.astype(np.int64).tobytes(), raw=True)
            for k, v in initializers.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
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

    print("fixtures written to", FIXTURES)


if __name__ == "__main__":
    main()
