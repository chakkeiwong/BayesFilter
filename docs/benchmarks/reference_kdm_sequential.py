"""Diagnostic-only CPU reference executing author sampling/weighting/KDE code."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path("/home/chakwong/BayesFilter")
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
sys.path.insert(0, str(AUTHOR / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
import torch
from kernel_density_estimation.kernel_density_estimator import KernelDensityEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    source_path = AUTHOR / "src/models/kde_particle_filter/kde_particle_filter.py"
    tree = ast.parse(source_path.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name in ("resample_particles", "weigh_particles")]
    assert len(methods) == 2
    namespace = dict(torch=torch, KernelDensityEstimator=KernelDensityEstimator)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), "exec"), namespace)
    settings = dict(batch=4, horizon=5, particles=25, sigma=.3, gamma=.4,
                    posterior_bandwidth=.5, resampling_bandwidth=.05,
                    density_epsilon=1e-8, injection_epsilon=1e-8,
                    initial_mean=.1, initial_variance=1.2, learning_rate=.001)
    actions = torch.tensor([[.2,-.1,.4,0.,-.2],[-.5,.3,.1,-.2,.6],
                            [.1,.2,-.3,.4,-.5],[0.,0.,.5,-.5,.2]])
    observations = torch.tensor([[.2,-.6,.8,-.3,.1],[1.1,-1.2,.3,.2,-.7],
                                 [-.2,.5,-.8,.9,-.4],[.7,-.5,.2,-.1,.6]])
    truth = torch.tensor([.3,-.4,.1,.7])
    original_sample = torch.distributions.Categorical.sample
    captured = []

    def record_sample(distribution, sample_shape=torch.Size()):
        result = original_sample(distribution, sample_shape)
        captured.append(result.transpose(0, 1).detach())
        return result

    results = {}
    try:
        torch.distributions.Categorical.sample = record_sample
        for mode in ("iwsg", "truncated"):
            torch.manual_seed(20260929)
            theta = torch.nn.Parameter(torch.tensor([.8,.4,1.,-1.2,.2,-.3,.4]))
            initial = .1 + math.sqrt(1.2) * torch.randn(4, 25)
            dynamics_noise = torch.randn(5, 4, 25)
            locations = initial.unsqueeze(-1)
            weights = torch.ones(4, 25) / 25
            bandwidth = torch.full((4, 1), .05)
            owner = types.SimpleNamespace(
                decouple_weights_for_resampling=False,
                decouple_bandwidths_for_resampling=False,
                use_differentiable_resampling=True, training=mode == "iwsg",
                differentiable_resampling_method="ImportanceSampling",
                weight_divide_by_proposal_probability=False,
                kde_params={"dims": {0: {"distribution_type": "Normal"}}},
                particle_transformer=types.SimpleNamespace(apply_norm=lambda x: x))
            rows = {key: [] for key in ("resampled", "particles", "weights",
                                       "ancestors", "uniforms", "kernel_noise")}
            for t in range(5):
                captured.clear()
                sample, injection, _ = namespace["resample_particles"](
                    owner, dict(particles_downscaled=locations, do_resample=True,
                                particle_weights=weights, bandwidths_downscaled=bandwidth))
                assert len(captured) == 1
                indices = captured[0]
                probability = weights.detach() / weights.detach().sum(1, keepdim=True)
                cdf = torch.cumsum(probability, dim=1)
                lower = torch.cat([torch.zeros(4, 1), cdf[:, :-1]], dim=1)
                midpoint = .5 * (cdf + lower)
                uniforms = midpoint.gather(1, indices)
                selected = locations.detach().squeeze(-1).gather(1, indices)
                kernel_noise = (sample.detach().squeeze(-1) - selected) / .05
                a, b, c1, c2, o1, o2, v = theta.unbind()
                cloud = a * sample.squeeze(-1) + b * actions[:, t, None] + .3 * dynamics_noise[t]
                means = torch.stack([c1 * cloud + o1, c2 * cloud + o2], dim=-1)
                residual = (observations[:, t, None, None] - means) / .4
                logcomp = -.5 * residual.square() - math.log(.4) - .5 * math.log(2 * math.pi)
                logmix = torch.stack([-torch.nn.functional.softplus(v),
                                      -torch.nn.functional.softplus(-v)])
                likelihood = torch.exp(torch.logsumexp(logcomp + logmix, dim=-1))
                raw_total = (likelihood * injection).sum(dim=1)
                assert bool((raw_total >= 1e-8).all())
                weights = namespace["weigh_particles"](
                    owner, {"final_input": cloud}, injection, torch.ones(4, 1),
                    lambda _: likelihood, {"internal_particles": cloud, "old_particles": locations})
                locations = cloud.unsqueeze(-1)
                values = dict(resampled=sample.squeeze(-1), particles=cloud, weights=weights,
                              ancestors=indices, uniforms=uniforms, kernel_noise=kernel_noise)
                for key, value in values.items():
                    rows[key].append(value.detach().tolist())
            density = KernelDensityEstimator(
                owner.kde_params, locations, weights, torch.full((4, 1), .5)).log_prob(
                    truth[:, None, None])[:, 0]
            loss = -density.mean()
            optimizer = torch.optim.SGD([theta], lr=.001, momentum=0., weight_decay=0.)
            optimizer.zero_grad()
            loss.backward()
            gradient = theta.grad.detach().tolist()
            before = theta.detach().tolist()
            optimizer.step()
            results[mode] = dict(
                **rows, theta=before, initial_particles=initial.tolist(), actions=actions.tolist(),
                observations=observations.tolist(), terminal_states=truth.tolist(),
                dynamics_noise=dynamics_noise.tolist(), terminal_logdensity=density.detach().tolist(),
                loss=float(loss.detach()), gradient=gradient, next_theta=theta.detach().tolist())
    finally:
        torch.distributions.Categorical.sample = original_sample
    kde_path = Path(sys.modules[KernelDensityEstimator.__module__].__file__)
    payload = dict(
        settings=settings, results=results, seed=20260929, torch=torch.__version__,
        cpu_only=True, gpu_intentionally_hidden=True, dtype="float64",
        role="CONSTRUCTED_ANALYTIC_CONSUMER_NOT_RECOVERED_AUTHOR_TRAINING_SETTINGS",
        source_methods=["resample_particles", "weigh_particles", "KernelDensityEstimator.sample",
                        "KernelDensityEstimator.log_prob"],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (source_path, kde_path)},
        sampler_replay="actual author sampler; recorded categories mapped to CDF interiors; realized Gaussian residuals")
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
