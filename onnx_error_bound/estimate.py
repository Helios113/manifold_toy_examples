"""Error bound of an ONNX graph: `estimate_bounds(model, inputs, profile)` runs the graph in float64 (the exact value of the program on the given float32 inputs and weights) while
propagating an absolute error bound through every node with the rules in rules.py and the calibrated backend profile. The bound is data dependent (it uses the actual values) and
covers the rounding of the runtime, not the model's own approximation error.

    from onnx_error_bound import estimate_bounds, load_profile
    r = estimate_bounds("models/toy_model.onnx", {"x": x}, load_profile("ort_cuda_fp32_h100"))
    r.outputs["logits"].v, r.outputs["logits"].e      # float64 reference, absolute error bound (same shape)
"""

from dataclasses import dataclass, field

import numpy as np
import onnx

from .profile import Profile, load_profile
from .rules import HANDLERS, SUPPORTED, T, Ctx, UnsupportedOp, attrs_of, const, tensor_to_array


@dataclass
class Result:
    outputs: dict
    warnings: list = field(default_factory=list)
    nodes: dict = field(default_factory=dict)  # every intermediate tensor, if requested


def estimate_bounds(model, inputs: dict, profile=None, input_rel_error: float = 0.0, keep_intermediates: bool = False) -> Result:
    """model: path or onnx.ModelProto. inputs: {name: float32 array}. profile: a Profile, a profile name/path, or None (uncalibrated textbook fp32).
    input_rel_error: relative uncertainty of the inputs (default 0 = the given float32 values are exact)."""
    m = onnx.load(model) if isinstance(model, str) else model
    P = profile if isinstance(profile, Profile) else load_profile(profile or "theory_fp32")
    ctx = Ctx(P)
    env = {}
    for init in m.graph.initializer:
        a = tensor_to_array(init)
        if np.issubdtype(a.dtype, np.floating):
            env[init.name] = const(a.astype(np.float32).astype(np.float64))
        else:
            env[init.name] = None
            ctx.consts[init.name] = a
    for vi in m.graph.input:
        if vi.name in env or vi.name in ctx.consts:
            continue
        if vi.name not in inputs:
            raise KeyError(f"missing input {vi.name!r}")
        x = np.asarray(inputs[vi.name]).astype(np.float32).astype(np.float64)
        env[vi.name] = T(x, np.abs(x) * input_rel_error)
    # shape/axes inputs of nodes are looked up by name; make them reachable through ctx.consts[id(...)] as the handlers expect
    for node in m.graph.node:
        if node.op_type == "Constant":
            val = tensor_to_array(attrs_of(node)["value"])
            if np.issubdtype(val.dtype, np.floating):
                env[node.output[0]] = const(val.astype(np.float32).astype(np.float64))
            else:
                env[node.output[0]] = None
                ctx.consts[node.output[0]] = val
            continue
        if node.op_type not in HANDLERS:
            raise UnsupportedOp(f"{node.op_type} (node {node.name!r}); supported: {', '.join(SUPPORTED)}")
        ins = []
        for name in node.input:
            if name == "":
                ins.append(None)
            elif env.get(name) is None and name in ctx.consts:  # int constant: a stand-in object carrying its name
                holder = _Const(name)
                ctx.consts[id(holder)] = ctx.consts[name]
                ins.append(holder)
            else:
                ins.append(env[name])
        outs = HANDLERS[node.op_type](ctx, attrs_of(node), ins)
        for name, t in zip(node.output, outs):
            env[name] = t
    outputs = {o.name: env[o.name] for o in m.graph.output}
    return Result(outputs=outputs, warnings=ctx.warnings, nodes=env if keep_intermediates else {})


class _Const:
    def __init__(self, name):
        self.name = name
