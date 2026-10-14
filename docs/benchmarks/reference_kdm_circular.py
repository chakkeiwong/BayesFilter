"""Independent CPU author consumer diagnostic; PyTorch/NumPy are reference only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
# A saved snapshot may be executed outside docs/benchmarks.
ROOT = Path("/home/chakwong/BayesFilter")
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
PACKAGE = Path("packages/kernel-density-estimator-bandwdidth-prediction/packages/kernel_density_estimation")
sys.path.insert(0, str(AUTHOR / PACKAGE.parent))
import torch
from kernel_density_estimation.kernel_density_estimator import KernelDensityEstimator
from kernel_density_estimation.kde_computer import KDEComputer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    hashes, anchors = {}, {}
    def verify(path):
        data = (AUTHOR / path).read_bytes()
        expected = subprocess.check_output(["git", "-C", str(AUTHOR), "show", PIN + ":" + str(path)])
        if data != expected:
            raise ValueError("Author source mismatch: " + str(path))
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data
    for filename in ("kernel_density_estimator.py", "kde_computer.py",
                     "von_mises_full_dist.py", "epanechnikov.py", "circular_epanechnikov.py"):
        verify(PACKAGE / filename)
    namespace = dict(torch=torch, KernelDensityEstimator=KernelDensityEstimator)
    def method(path, class_name, name, ns):
        tree = ast.parse(verify(path))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
        node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        anchors[str(path) + "::" + name] = [node.lineno, node.end_lineno]
        return ns[name]
    model_path = Path("src/models/kde_particle_filter/kde_particle_filter.py")
    resample = method(model_path, "KDEParticleFilter", "resample_particles", namespace)
    weigh = method(model_path, "KDEParticleFilter", "weigh_particles", namespace)
    apply_norm = method(Path("src/models/particle_transformer.py"),
                        "BearingsOnlyTransformer", "apply_norm", dict(torch=torch))
    loss_method = method(Path("src/loss_functions/loss_functions.py"),
                         "KDE_NLL", "compute_loss", dict(torch=torch))
    params = {"dims": {0: {"distribution_type": "Normal"},
                        1: {"distribution_type": "Normal"},
                        2: {"distribution_type": "Von_Mises"}}}
    transformer = SimpleNamespace(apply_norm=lambda x: apply_norm(None, x))
    owner = SimpleNamespace(kde_params=params, decouple_weights_for_resampling=False,
                            decouple_bandwidths_for_resampling=False, training=True,
                            use_differentiable_resampling=True,
                            differentiable_resampling_method="ImportanceSampling",
                            particle_transformer=transformer, weight_divide_by_proposal_probability=False)
    loss_owner = SimpleNamespace(manual_bandwidth=None, kde_computer=KDEComputer(params))
    cases = []
    def host(x):
        if isinstance(x, (list, tuple)):
            return [host(v) for v in x]
        return x.detach().tolist()
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        torch.set_default_dtype(dtype)
        for case_index, name in enumerate(("ordinary", "angular_seam", "tiny_weights", "far_tail")):
            torch.manual_seed(20261006 + case_index)
            u = torch.arange(50, dtype=dtype).reshape(2, 25)
            loc = torch.stack([.7 * torch.sin(u), .5 * torch.cos(u * .7),
                               (u % 2 * 2 - 1) * (math.pi - .02)], -1)
            weight = 1. + (u % 5)
            weight /= weight.sum(-1, keepdim=True)
            band = torch.tensor([[.4, .8, 4.], [.6, .35, .25]])
            if name == "angular_seam":
                band[:, 2] = torch.tensor([.01, .04])
            if name == "tiny_weights":
                weight[:, 0] = 0.
                weight *= 1e-10
            states = torch.tensor([[-.4, .3, math.pi - .01], [.2, -.4, -math.pi + .01]])
            if name == "far_tail":
                states[:, :2] += 50.
            loc, weight, band = [x.detach().requires_grad_(True) for x in (loc, weight, band)]
            samples, iwsg, dynamics_weight = resample(owner, dict(
                particles_downscaled=loc, particle_weights=weight,
                bandwidths_downscaled=band, do_resample=True))
            density = KernelDensityEstimator(params, loc, weight, band).log_prob(samples)
            integrand = (samples.square() + .3 * samples).sum(-1)
            raw = (iwsg * integrand).sum()
            normalized = (iwsg / iwsg.sum(-1, keepdim=True) * integrand).sum()
            raw_gradient = torch.autograd.grad(raw, [loc, weight, band], retain_graph=True)
            normalized_gradient = torch.autograd.grad(normalized, [loc, weight, band], retain_graph=True)
            direct_nll = loss_method(loss_owner, dict(particles=loc, particle_weights=weight,
                                                      bandwidths=band), states).squeeze(-1)
            direct_density = loss_owner.kde_computer.compute_kde(
                states.unsqueeze(1), loc, weight, band).squeeze(-1)
            direct_loss = direct_nll.mean()
            direct_gradient = torch.autograd.grad(direct_loss, [loc, weight, band])
            posterior_loc = (samples + torch.tensor([.1, -.1, .02])).detach().requires_grad_(True)
            model_weight = (.2 + torch.cos(u).square()).detach().requires_grad_(True)
            posterior_band = torch.tensor([[.5, .6, .2], [.4, .7, .4]], requires_grad=True)
            normalized_weight = weigh(owner, {"final_input": posterior_loc}, iwsg, dynamics_weight,
                                      lambda inputs: model_weight, {})
            consumer_nll = loss_method(loss_owner, dict(
                particles=posterior_loc, particle_weights=normalized_weight,
                bandwidths=posterior_band), states).squeeze(-1)
            consumer_loss = consumer_nll.mean()
            consumer_gradient = torch.autograd.grad(
                consumer_loss, [loc, weight, band, posterior_loc, model_weight, posterior_band])
            wrong_log = KernelDensityEstimator(params, loc, weight, band).log_prob(states.unsqueeze(1))
            wrong_nll = -torch.log(torch.exp(wrong_log.squeeze(-1)) + 1e-8)
            cases.append(dict(
                id=dtype_name + "_" + name, dtype=dtype_name, regime=name,
                inputs=dict(locations=host(loc), weights=host(weight), bandwidth=host(band),
                            samples=host(samples), states=host(states),
                            posterior_locations=host(posterior_loc), model_weights=host(model_weight),
                            posterior_bandwidth=host(posterior_band)),
                resampling=dict(log_density=host(density), iwsg=host(iwsg),
                                raw_loss=host(raw), normalized_loss=host(normalized),
                                raw_gradient=host(raw_gradient), normalized_gradient=host(normalized_gradient),
                                sample_gradient_stopped=not samples.requires_grad),
                posterior=dict(density=host(direct_density), nll=host(direct_nll), loss=host(direct_loss),
                               gradient=host(direct_gradient)),
                consumer=dict(nll=host(consumer_nll), loss=host(consumer_loss),
                              normalized_weights=host(normalized_weight), gradient=host(consumer_gradient)),
                wrong_density_substitution_max_error=float((wrong_nll - direct_nll).detach().abs().max())))
    vm_path = Path(inspect.getfile(torch.distributions.VonMises))
    args.output.write_text(json.dumps(dict(
        schema="kdm_circular_author_reference_v1", cpu_only=True,
        gpu_intentionally_hidden=True, torch=torch.__version__, author_commit=PIN,
        source_hashes=hashes, source_anchors=anchors,
        von_mises_implementation=str(vm_path),
        von_mises_sha256=hashlib.sha256(vm_path.read_bytes()).hexdigest(),
        seed_policy="20261006 + regime index, repeated independently for each dtype",
        fixture="constructed mechanics; not training data", cases=cases),
        indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="PASS", source_cases=len(cases), cpu_only=True)))


if __name__ == "__main__":
    main()
