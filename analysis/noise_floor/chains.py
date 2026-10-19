"""Chained-operator test programs for the noise-floor checks (c18), written once in a tiny IR and run three ways: by torch (c18_torch), as an ONNX graph in ONNX Runtime (c18_onnx),
and by the float64 reference + error tracer here (c18_analyse). numpy only, so it imports in both environments (the ONNX one has no torch).

IR: every leaf (data, weights, scalar constants) is a graph input; nodes are (out, kind, inputs) with kind in add/sub/mul/div/sqrt/erf/matmul/sum (sum = over the last axis, keepdims).
The tracer carries an absolute error bound per node with the standard-model rules (u = 2^-24, k_f = rounding-term multiplier measured for the non-correctly-rounded ops)."""

import numpy as np

U32 = 2.0**-24
B, D, H = 4096, 16, 64  # batch, model width, MLP hidden width (as in this repo's ResMLPBlock)


class Chain:
    def __init__(self, name):
        self.name, self.nodes, self.inputs, self.shapes, self.sampler, self.outputs = name, [], [], {}, {}, {}

    def inp(self, name, shape, sampler):
        self.inputs.append(name)
        self.shapes[name], self.sampler[name] = shape, sampler
        return name

    def op(self, kind, *ins):
        out = f"v{len(self.nodes)}"
        self.nodes.append((out, kind, ins))
        return out


normal = lambda s=1.0: (lambda rng, shape: rng.standard_normal(shape) * s)
uniform = lambda a, b: (lambda rng, shape: rng.uniform(a, b, shape))
const = lambda v: (lambda rng, shape: np.full(shape, v))


def layernorm(c, x, d=D):
    inv_d, eps = c.inp("inv_d", (), const(1.0 / d)), c.inp("eps", (), const(1e-5))
    m = c.op("mul", c.op("sum", x), inv_d)
    cc = c.op("sub", x, m)
    v = c.op("mul", c.op("sum", c.op("mul", cc, cc)), inv_d)
    return c.op("div", cc, c.op("sqrt", c.op("add", v, eps)))


def gelu(c, x):
    half, one, r2 = c.inp("half", (), const(0.5)), c.inp("one", (), const(1.0)), c.inp("inv_sqrt2", (), const(2.0**-0.5))
    a = c.op("erf", c.op("mul", x, r2))
    return c.op("mul", c.op("mul", x, half), c.op("add", a, one))


def affine3():
    c = Chain("affine3")
    h = c.inp("x", (B, D), normal())
    for i in range(3):
        h = c.op("matmul", h, c.inp(f"W{i}", (D, D), normal(D**-0.5)))
    c.outputs = {"y": h}
    return c


def layernorm_chain():
    c = Chain("layernorm")
    c.outputs = {"y": layernorm(c, c.inp("x", (B, D), normal()))}
    return c


def gelu_chain():
    c = Chain("gelu")
    c.outputs = {"y": gelu(c, c.inp("x", (B, D), normal(2.0)))}
    return c


def block(b=B, d=D, h=H):
    c = Chain("block")  # x + W2 gelu(W1 LN(x) + b1) + b2: the forward pass of this repo's ResMLPBlock (LayerNorm without affine); sizes (batch, width, hidden) are parameters for the speed test
    x = c.inp("x", (b, d), normal())
    hh = c.op("add", c.op("matmul", layernorm(c, x, d), c.inp("W1", (d, h), normal(d**-0.5))), c.inp("b1", (h,), normal(0.1)))
    o = c.op("add", c.op("matmul", gelu(c, hh), c.inp("W2", (h, d), normal(h**-0.5))), c.inp("b2", (d,), normal(0.1)))
    c.outputs = {"y": c.op("add", x, o)}
    return c


def sqrt_chain(depth=16):
    c = Chain("sqrt_chain")
    h = c.inp("x", (B, D), uniform(0.5, 2.0))
    for i in range(1, depth + 1):
        h = c.op("sqrt", h)
        if i in (1, 2, 4, 8, 16):
            c.outputs[f"after_{i}"] = h
    return c


def prod_chain(m=64):
    c = Chain("prod_chain")
    h = c.inp("x", (B, D), uniform(0.5, 2.0))
    for i in range(1, m + 1):
        h = c.op("mul", h, c.inp(f"c{i}", (), uniform(0.9, 1.1)))
        if i in (4, 16, 64):
            c.outputs[f"after_{i}"] = h
    return c


N_CAL = 1_000_000


def cal_erf():  # single-op calibration chains: how accurate is one erf / sqrt / div in this backend
    c = Chain("cal_erf")
    c.outputs = {"y": c.op("erf", c.inp("x", (N_CAL, 1), uniform(-4, 4)))}
    return c


def cal_sqrt():
    c = Chain("cal_sqrt")
    c.outputs = {"y": c.op("sqrt", c.inp("x", (N_CAL, 1), uniform(1e-3, 1e3)))}
    return c


def cal_div():
    c = Chain("cal_div")
    c.outputs = {"y": c.op("div", c.inp("a", (N_CAL, 1), uniform(0.1, 10)), c.inp("b", (N_CAL, 1), uniform(0.1, 10)))}
    return c


N_MM = 65536


def cal_mm16():  # one matmul with inner dimension 16 / 64 (the sizes of the block): worst |error| / sum|a_i b_i| over all entries
    c = Chain("cal_mm16")
    c.outputs = {"y": c.op("matmul", c.inp("x", (N_MM, 16), normal()), c.inp("W", (16, 16), normal()))}
    return c


def cal_mm64():
    c = Chain("cal_mm64")
    c.outputs = {"y": c.op("matmul", c.inp("x", (N_MM, 64), normal()), c.inp("W", (64, 16), normal()))}
    return c


TEST = {f.__name__: f for f in [affine3, layernorm_chain, gelu_chain, block, sqrt_chain, prod_chain]}
CAL = {f.__name__: f for f in [cal_erf, cal_sqrt, cal_div, cal_mm16, cal_mm64]}
CHAINS = {**TEST, **CAL}


def make_inputs(c: Chain, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return {n: np.asarray(c.sampler[n](rng, c.shapes[n]), dtype=np.float32) for n in c.inputs}  # float32: the exact values every backend receives


def out_shapes(c: Chain) -> dict:
    """Shape of every node (numpy only; the ONNX graph outputs need explicit shapes)."""
    sh = dict(c.shapes)
    for out, kind, ins in c.nodes:
        a = [sh[i] for i in ins]
        sh[out] = {"matmul": lambda: a[0][:-1] + a[1][-1:], "sum": lambda: a[0][:-1] + (1,), "sqrt": lambda: a[0], "erf": lambda: a[0]}.get(kind, lambda: np.broadcast_shapes(*a))()
    return sh


def eval_torch(c: Chain, inp: dict, device: str) -> dict:
    import torch

    f = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b, "matmul": torch.matmul,
         "sum": lambda a: a.sum(-1, keepdim=True), "sqrt": torch.sqrt, "erf": torch.erf}
    env = {n: torch.tensor(inp[n], device=device) for n in c.inputs}
    with torch.no_grad():
        for out, kind, ins in c.nodes:
            env[out] = f[kind](*[env[i] for i in ins])
    return {lab: env[n].cpu().numpy() for lab, n in c.outputs.items()}


def eval_numpy(c: Chain, inp: dict) -> dict:
    """float64 reference of the same program on the same float32 inputs; returns every node's value."""
    from scipy.special import erf

    f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "matmul": np.matmul, "sum": lambda a: a.sum(-1, keepdims=True), "sqrt": np.sqrt, "erf": erf}
    env = {n: inp[n].astype(np.float64) for n in c.inputs}
    for out, kind, ins in c.nodes:
        env[out] = f[kind](*[env[i] for i in ins])
    return env


def gamma(n):
    return n * U32 / (1 - n * U32)


def trace(c: Chain, vals: dict, k: dict) -> dict:
    """Absolute error bound of every node. vals = eval_numpy(...). k = rounding-term multipliers for sqrt/div/erf (1 for a correctly rounded op); add/sub/mul use u,
    matmul and sum use gamma_n (valid for any summation order)."""
    e = {n: np.zeros_like(vals[n]) for n in c.inputs}
    tiny = np.finfo(np.float64).tiny
    for out, kind, ins in c.nodes:
        v, a, ea = vals[out], [vals[i] for i in ins], [e[i] for i in ins]
        kk = 1.0
        if kind in ("add", "sub"):
            pre = ea[0] + ea[1]
        elif kind == "mul":
            pre = np.abs(a[0]) * ea[1] + np.abs(a[1]) * ea[0] + ea[0] * ea[1]
        elif kind == "div":
            den = np.abs(a[1])
            pre, kk = (ea[0] * den + np.abs(a[0]) * ea[1]) / (den * np.maximum(den - ea[1], tiny)), k["div"]
        elif kind == "sqrt":
            pre, kk = ea[0] / (np.sqrt(a[0]) + np.sqrt(np.maximum(a[0] - ea[0], 0))), k["sqrt"]
        elif kind == "erf":
            pre, kk = 2 / np.sqrt(np.pi) * np.exp(-np.maximum(np.abs(a[0]) - ea[0], 0) ** 2) * ea[0], k["erf"]
        elif kind == "matmul":
            A, Bm = np.abs(a[0]), np.abs(a[1])
            n = a[0].shape[-1]
            if "mm_coef" in k:  # reduced-precision matmul (TF32): relative coefficient on sum|a b| of the computed operands, e.g. 2 u_tf32 + gamma_n
                mags = (A + ea[0]) @ (Bm + ea[1]) * k["mm_coef"](n)
            else:
                mags = gamma(n) * (A @ Bm)
            pre, kk = A @ ea[1] + ea[0] @ Bm + ea[0] @ ea[1] + mags, 0.0
        elif kind == "sum":
            pre, kk = ea[0].sum(-1, keepdims=True) + gamma(a[0].shape[-1] - 1) * np.abs(a[0]).sum(-1, keepdims=True), 0.0
        e[out] = pre + kk * U32 * (np.abs(v) + pre)
    return e
