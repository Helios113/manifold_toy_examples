"""Per-operator error propagation rules. Every tensor is carried as T(v, e): v = the exact (float64) value of the program run on the float32 inputs and weights, e = an absolute bound on
|runtime result - v|. A rule computes the output (v, e) from the inputs' (v, e): first the error propagated from the inputs, then the rounding error of the operation itself
(u |result| * k for elementwise ops; gamma_n sum|a b| for dot products and sums, valid for any summation order). The profile supplies u, k and the matmul coefficient."""

from dataclasses import dataclass
from math import pi, prod

import numpy as np
from onnx import helper, numpy_helper
from scipy.special import erf as _erf, expit

TINY = np.finfo(np.float64).tiny


class UnsupportedOp(Exception):
    pass


@dataclass
class T:
    v: np.ndarray
    e: np.ndarray


def const(x) -> T:
    x = np.asarray(x, dtype=np.float64)
    return T(x, np.zeros_like(x))


class Ctx:
    """Per-estimate state: the profile, int constants (shapes, axes), and warnings."""

    def __init__(self, profile):
        self.P, self.consts, self.warnings = profile, {}, []

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def check_range(self, op, x: T):
        r = self.P.ranges.get(op)
        if r and x.v.size and (x.v.min() < r[0] or x.v.max() > r[1]):
            self.warn(f"{op}: input range [{x.v.min():.3g}, {x.v.max():.3g}] is outside the calibrated range {r}")


def _round(ctx, v, pre, op=None):
    return pre + ctx.P.k.get(op, 1.0) * ctx.P.u * (np.abs(v) + pre)


def add(ctx, a, b):
    v = a.v + b.v
    return T(v, _round(ctx, v, a.e + b.e))


def sub(ctx, a, b):
    v = a.v - b.v
    return T(v, _round(ctx, v, a.e + b.e))


def mul(ctx, a, b):
    v = a.v * b.v
    return T(v, _round(ctx, v, np.abs(a.v) * b.e + np.abs(b.v) * a.e + a.e * b.e))


def div(ctx, a, b):
    v = a.v / b.v
    den = np.abs(b.v)
    return T(v, _round(ctx, v, (a.e * den + np.abs(a.v) * b.e) / (den * np.maximum(den - b.e, TINY)), "Div"))


def div_const(ctx, a, n):  # division by an exact integer
    v = a.v / n
    return T(v, _round(ctx, v, a.e / n))


def sqrt(ctx, a):
    v = np.sqrt(a.v)
    return T(v, _round(ctx, v, a.e / (np.sqrt(a.v) + np.sqrt(np.maximum(a.v - a.e, 0))), "Sqrt"))


def reciprocal(ctx, a):
    return div(ctx, const(np.ones_like(a.v)), a)


def exp(ctx, a):
    ctx.check_range("Exp", a)
    v = np.exp(a.v)
    return T(v, _round(ctx, v, v * np.expm1(a.e), "Exp"))


def log(ctx, a):
    ctx.check_range("Log", a)
    v = np.log(a.v)
    return T(v, _round(ctx, v, a.e / np.maximum(a.v - a.e, TINY), "Log"))


def erf(ctx, a):
    ctx.check_range("Erf", a)
    v = _erf(a.v)
    return T(v, _round(ctx, v, 2 / np.sqrt(pi) * np.exp(-np.maximum(np.abs(a.v) - a.e, 0) ** 2) * a.e, "Erf"))


def tanh(ctx, a):
    ctx.check_range("Tanh", a)
    v = np.tanh(a.v)
    return T(v, _round(ctx, v, a.e * (1 - np.tanh(np.maximum(np.abs(a.v) - a.e, 0)) ** 2), "Tanh"))


def sigmoid(ctx, a):
    ctx.check_range("Sigmoid", a)
    v = expit(a.v)
    s = expit(np.maximum(np.abs(a.v) - a.e, 0))
    return T(v, _round(ctx, v, a.e * s * (1 - s), "Sigmoid"))


def reduce_sum(ctx, a, axes, keepdims):
    n = prod(a.v.shape[i] for i in axes)
    u = ctx.P.u
    gam = (n - 1) * u / (1 - (n - 1) * u)
    return T(a.v.sum(axes, keepdims=keepdims), a.e.sum(axes, keepdims=keepdims) + gam * np.abs(a.v).sum(axes, keepdims=keepdims))


def matmul(ctx, a, b):
    A, B = np.abs(a.v), np.abs(b.v)
    coef = ctx.P.matmul_coef(a.v.shape[-1])
    return T(a.v @ b.v, A @ b.e + a.e @ B + a.e @ b.e + coef * ((A + a.e) @ (B + b.e)))


# --- ONNX node handlers: h(ctx, attrs, ins) -> [T, ...] -------------------------------------------------------------------------------------------------------------------
def _axes(ctx, attrs, ins, ndim):
    ax = ctx.consts.get(id(ins[1])) if len(ins) > 1 and ins[1] is not None else attrs.get("axes")
    return tuple(range(ndim)) if ax is None else tuple(int(i) % ndim for i in np.atleast_1d(ax))


def h_reduce(kind):
    def h(ctx, attrs, ins):
        x = ins[0]
        axes = _axes(ctx, attrs, ins, x.v.ndim)
        keep = bool(attrs.get("keepdims", 1))
        s = reduce_sum(ctx, x, axes, keep)
        return [s if kind == "sum" else div_const(ctx, s, prod(x.v.shape[i] for i in axes))]

    return h


def h_gemm(ctx, attrs, ins):
    a, b = ins[0], ins[1]
    if attrs.get("transA", 0):
        a = T(a.v.T, a.e.T)
    if attrs.get("transB", 0):
        b = T(b.v.T, b.e.T)
    y = matmul(ctx, a, b)
    if attrs.get("alpha", 1.0) != 1.0:
        y = mul(ctx, y, const(np.float32(attrs["alpha"])))
    if len(ins) > 2 and ins[2] is not None:
        c = ins[2]
        if attrs.get("beta", 1.0) != 1.0:
            c = mul(ctx, c, const(np.float32(attrs["beta"])))
        y = add(ctx, y, c)
    return [y]


def h_layernorm(ctx, attrs, ins):
    x = ins[0]
    scale, bias = ins[1], (ins[2] if len(ins) > 2 else None)
    axes = tuple(range(attrs.get("axis", -1) % x.v.ndim, x.v.ndim))
    n = prod(x.v.shape[i] for i in axes)
    m = div_const(ctx, reduce_sum(ctx, x, axes, True), n)
    c = sub(ctx, x, m)
    var = div_const(ctx, reduce_sum(ctx, mul(ctx, c, c), axes, True), n)
    sg = sqrt(ctx, add(ctx, var, const(np.float32(attrs.get("epsilon", 1e-5)))))
    y = mul(ctx, div(ctx, c, sg), scale)
    if bias is not None:
        y = add(ctx, y, bias)
    f = ctx.P.composite.get("LayerNormalization", 1.0)
    return [T(y.v, y.e * f)]


def h_gelu(ctx, attrs, ins):
    if attrs.get("approximate", b"none") not in (b"none", "none"):
        raise UnsupportedOp("Gelu with approximate='tanh'")
    x = ins[0]
    ctx.check_range("Gelu", x)
    r2 = 2.0**-0.5  # the kernel multiplies by the float32 constant 1/sqrt(2): relative error <= u
    t = mul(ctx, x, T(np.float64(r2), np.float64(ctx.P.u * r2)))
    b = add(ctx, erf(ctx, t), const(1.0))
    y = mul(ctx, mul(ctx, x, const(0.5)), b)
    return [T(y.v, y.e * ctx.P.composite.get("Gelu", 1.0))]


def _shape_op(fn):
    def h(ctx, attrs, ins):
        return [T(fn(ctx, attrs, ins, ins[0].v), fn(ctx, attrs, ins, ins[0].e))]

    return h


def _reshape(ctx, attrs, ins, x):
    return x.reshape([int(s) if s != 0 else x.shape[i] for i, s in enumerate(ctx.consts[id(ins[1])])])


def _unsq(ctx, attrs, ins, x):
    for a in sorted(int(i) for i in np.atleast_1d(ctx.consts.get(id(ins[1])) if len(ins) > 1 else attrs["axes"])):
        x = np.expand_dims(x, a)
    return x


def _sq(ctx, attrs, ins, x):
    ax = ctx.consts.get(id(ins[1])) if len(ins) > 1 else attrs.get("axes")
    return np.squeeze(x, None if ax is None else tuple(int(i) for i in np.atleast_1d(ax)))


UNARY = {"Sqrt": sqrt, "Exp": exp, "Log": log, "Erf": erf, "Tanh": tanh, "Sigmoid": sigmoid, "Reciprocal": reciprocal}
HANDLERS = {
    "Add": lambda c, a, i: [add(c, i[0], i[1])], "Sub": lambda c, a, i: [sub(c, i[0], i[1])], "Mul": lambda c, a, i: [mul(c, i[0], i[1])], "Div": lambda c, a, i: [div(c, i[0], i[1])],
    "Neg": lambda c, a, i: [T(-i[0].v, i[0].e)], "Abs": lambda c, a, i: [T(np.abs(i[0].v), i[0].e)], "Relu": lambda c, a, i: [T(np.maximum(i[0].v, 0), i[0].e)],  # exact, 1-Lipschitz
    "Identity": lambda c, a, i: [i[0]],
    "MatMul": lambda c, a, i: [matmul(c, i[0], i[1])], "Gemm": h_gemm, "LayerNormalization": h_layernorm, "Gelu": h_gelu,
    "ReduceSum": h_reduce("sum"), "ReduceMean": h_reduce("mean"),
    "Reshape": _shape_op(_reshape), "Flatten": _shape_op(lambda c, a, i, x: x.reshape(prod(x.shape[: a.get("axis", 1)]), -1)),
    "Transpose": _shape_op(lambda c, a, i, x: np.transpose(x, a.get("perm"))), "Unsqueeze": _shape_op(_unsq), "Squeeze": _shape_op(_sq),
    **{k: (lambda f: lambda c, a, i: [f(c, i[0])])(f) for k, f in UNARY.items()},
}
SUPPORTED = sorted(HANDLERS)


def attrs_of(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def tensor_to_array(t):
    return numpy_helper.to_array(t)
