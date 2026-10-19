"""Check 00: the models are real. Weights finite, accuracy on the sphere, analytic Jacobian == autograd Jacobian, ONNX graph == torch model (and how far fp32 is from fp64)."""

import onnxruntime

from analysis.noise_floor.deferred_models.model_common import *

N = 6000
p = sample(N)
x, y = man.embed(p), man.labels(p)
models = load_models()
info, jac_err = {}, {}
for name, m in models.items():
    with torch.no_grad():
        acc = (m(x.double()).argmax(-1) == y).double().mean().item()
    w = torch.cat([q.flatten() for q in m.parameters()])
    info[name] = dict(n_params=w.numel(), all_finite=bool(torch.isfinite(w).all()), weight_norm=w.norm().item(), accuracy_on_sphere=acc)
    for b, blk, h, J in jacobians(m, p[:40]):
        errs = []
        for i in range(len(h)):
            Ja = torch.autograd.functional.jacobian(lambda z: blk(z[None])[0], h[i].clone())
            errs.append(((J[i] - Ja).abs().max() / J[i].abs().max()).item())
        jac_err[f"{name}/block{b}"] = errs
        info[name][f"max_rel_diff_analytic_vs_autograd_J_block{b}"] = max(errs)

m64 = models["trained"]
m32 = copy.deepcopy(m64).float()
sess = onnxruntime.InferenceSession("models/toy_model.onnx")
lo_onnx = sess.run(None, {"x": x.float().numpy()})[0].astype(np.float64)
with torch.no_grad():
    lo32, lo64 = m32(x.float()).double().numpy(), m64(x.double()).numpy()
scale = np.abs(lo64).max()
d = dict(onnx_vs_torch32=np.abs(lo_onnx - lo32).max(), onnx_vs_torch64=np.abs(lo_onnx - lo64).max(), torch32_vs_torch64=np.abs(lo32 - lo64).max())
info["logit_max_abs_diff"] = d
info["logit_scale_max_abs"] = scale
info["onnx_argmax_agrees_with_torch64"] = float((lo_onnx.argmax(-1) == lo64.argmax(-1)).mean())
ok = all(v["all_finite"] for k, v in info.items() if k in models) and max(max(e) for e in jac_err.values()) < 1e-10 and d["onnx_vs_torch64"] < 1e-4 * scale
result("c00_models", "weights are finite, analytic J == autograd J (float64), ONNX == torch within fp32 noise", info, "holds" if ok else "violated")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].bar(list(models), [info[k]["accuracy_on_sphere"] for k in models], color="#2a78d6")
ax[0].set(ylabel=f"accuracy on {N} sphere points (float64)", ylim=(0, 1.02), title="models")
for k, e in jac_err.items():
    ax[1].plot(sorted(e), label=k)
ax[1].set(yscale="log", xlabel="point (sorted)", ylabel="max|J_analytic - J_autograd| / max|J|", title="analytic vs autograd Jacobian (float64)")
ax[1].legend(fontsize=6)
ax[2].bar(list(d), list(d.values()), color="#d97a1a")
ax[2].set(yscale="log", ylabel="max |logit difference|", title="trained model: ONNX / fp32 / fp64 logits")
ax[2].tick_params(axis="x", labelsize=7)
save(fig, "c00_models")
