"""ONNX Runtime helpers shared by calibrate.py and validate.py (the only modules that need onnxruntime)."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from .profile import Profile


def session(model, profile: Profile) -> "ort.InferenceSession":
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if profile.opt_level == "all" else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    providers = [(p, profile.provider_options[p]) if p in profile.provider_options else p for p in profile.providers]
    s = ort.InferenceSession(model if isinstance(model, (bytes, str)) else model.SerializeToString(), so, providers=providers)
    if profile.providers[0] != "CPUExecutionProvider" and s.get_providers()[0] != profile.providers[0]:
        raise RuntimeError(f"{profile.providers[0]} requested but the session uses {s.get_providers()}")
    return s


def run(model, profile: Profile, inputs: dict) -> list:
    s = session(model, profile)
    return s.run(None, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in inputs.items()})


def single_op_model(op, in_shapes: dict, out_shape, **attrs) -> bytes:
    """A graph with one ONNX node (opset 20); every operand is a graph input."""
    node = helper.make_node(op, list(in_shapes), ["y"], **attrs)
    g = helper.make_graph([node], f"cal_{op}", [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(s)) for n, s in in_shapes.items()],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m.SerializeToString()
