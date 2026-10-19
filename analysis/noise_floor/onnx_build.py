"""ONNX graph builder for the chain IR (opset 17, one ONNX node per IR node). Shared by c18_onnx.py and c19_onnx.py."""

import onnx
from onnx import TensorProto, helper

from analysis.noise_floor.chains import out_shapes

OP = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div", "matmul": "MatMul", "sqrt": "Sqrt", "erf": "Erf"}


def build(c, inp):
    nodes, init = [], []
    for out, kind, ins in c.nodes:
        if kind == "sum":
            if not init:
                init.append(helper.make_tensor("axes_last", TensorProto.INT64, [1], [-1]))
            nodes.append(helper.make_node("ReduceSum", [ins[0], "axes_last"], [out], keepdims=1))
        else:
            nodes.append(helper.make_node(OP[kind], list(ins), [out]))
    g = helper.make_graph(nodes, c.name, [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(inp[n].shape)) for n in c.inputs],
                          [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(out_shapes(c)[n])) for n in c.outputs.values()], initializer=init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m.SerializeToString()
