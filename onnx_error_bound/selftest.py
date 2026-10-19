"""Runtime-free self test (numpy/scipy/onnx only): builds a small ONNX graph (Gemm, LayerNormalization, Gelu, Add, Reshape, ReduceMean) in memory, evaluates it in float32 with plain numpy
and checks the estimated bound (uncalibrated textbook profile) is not exceeded and an unsupported op raises. python -m onnx_error_bound.selftest"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.special import erf

from . import UnsupportedOp, estimate_bounds


def graph(w1, b1, ln_s, ln_b, w2):
    f = lambda n, a: numpy_helper.from_array(a.astype(np.float32), n)
    nodes = [
        helper.make_node("Gemm", ["x", "w1", "b1"], ["h"], transB=1),
        helper.make_node("LayerNormalization", ["h", "ln_s", "ln_b"], ["n"], axis=-1, epsilon=1e-5),
        helper.make_node("Gelu", ["n"], ["g"], approximate="none"),
        helper.make_node("MatMul", ["g", "w2"], ["m"]),
        helper.make_node("Add", ["m", "x"], ["y"]),
        helper.make_node("ReduceMean", ["y"], ["z"], axes=[1], keepdims=0),
    ]
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 16])], [helper.make_tensor_value_info("z", TensorProto.FLOAT, [None])],
                          [f("w1", w1), f("b1", b1), f("ln_s", ln_s), f("ln_b", ln_b), f("w2", w2)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
    m.ir_version = 9
    return m


def main():
    rng = np.random.default_rng(0)
    w1, b1, s, b, w2 = (rng.standard_normal(sh) * sc for sh, sc in [((16, 16), 0.25), (16, 0.1), (16, 1), (16, 0.1), ((16, 16), 0.25)])
    m = graph(w1, b1, s, b, w2)
    x = rng.standard_normal((2000, 16)).astype(np.float32)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    h = f32(x @ f32(w1).T) + f32(b1)
    mu = h.mean(-1, keepdims=True, dtype=np.float32)
    c = h - mu
    var = (c * c).mean(-1, keepdims=True, dtype=np.float32)
    n = c / np.sqrt(var + np.float32(1e-5), dtype=np.float32) * f32(s) + f32(b)
    g = f32(0.5) * n * (f32(1) + f32(erf(n * np.float32(2**-0.5))))
    y = f32(g @ f32(w2)) + x
    z = y.mean(-1, dtype=np.float32)
    r = estimate_bounds(m, {"x": x})
    zt = r.outputs["z"]
    ratio = np.abs(z.astype(np.float64) - zt.v) / zt.e
    print(f"selftest: max error / bound = {ratio.max():.3f}, median bound / error = {np.median(zt.e / np.maximum(np.abs(z - zt.v), 1e-30)):.0f}")
    assert ratio.max() <= 1.0, "bound exceeded by a plain float32 numpy implementation"
    bad = helper.make_model(helper.make_graph([helper.make_node("Softmax", ["x"], ["y"])], "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
                                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])]), opset_imports=[helper.make_opsetid("", 20)])
    try:
        estimate_bounds(bad, {"x": np.zeros((1, 4), np.float32)})
        raise AssertionError("an unsupported op must raise")
    except UnsupportedOp as e:
        print("selftest: unsupported op raises:", str(e)[:60])
    print("selftest passed")


if __name__ == "__main__":
    main()
