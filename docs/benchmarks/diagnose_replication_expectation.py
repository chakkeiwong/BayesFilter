"""Independent CPU reference diagnostic of IWSG expectation; not a runtime implementation."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU diagnostic must hide GPUs before framework import")
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    spec = importlib.util.spec_from_file_location(
        "reference_training", Path(__file__).with_name("diagnose_replication_training.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config = helper.restore_yaml_dimensions(json.loads(args.configuration.read_text()))
    src = helper.source_copy(args.author_root, args.output, torch)
    sys.path.insert(0, str(args.author_root / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
    sys.path.insert(0, str(src))
    from models.model_creator import create_model
    params = copy.deepcopy(next(e["parameters"] for e in config["experiments"]
                               if e["name"] == "full_dpf_full_training_fixed_bands_000"))
    torch.manual_seed(924150)
    model = create_model(params["model_params"], config["architecture"]).double().train()
    assert model.differentiable_resampling_method == "ImportanceSampling"
    assert not model.decouple_weights_for_resampling and not model.decouple_bandwidths_for_resampling
    assert [v["distribution_type"] for v in model.kde_params["dims"].values()] == ["Normal", "Normal", "Von_Mises"]
    epsilon, injected_constant = 1e-8, 1 + 1e-8
    seeds = list(range(924151, 924159))
    batch = 1024
    contract = {
        "purpose": "one-step expectation for diagnostic linear/constant statistics",
        "CPU_only": True, "CUDA_VISIBLE_DEVICES": "-1", "dtype": "float64",
        "model_initialization_seed": 924150, "sampling_seeds": seeds,
        "clouds_per_seed": batch, "particle_counts": [2, 25],
        "quadrature_orders": [64, 128], "quadrature_atol": 1e-13, "quadrature_rtol": 1e-7,
        "algebra_atol": 1e-10, "algebra_rtol": 1e-8,
        "expectation_tolerance": "6 MCSE + quadrature difference + 1e-12",
        "interval": "approximate normal 95% interval using independent-cloud sample variance",
        "rare_case": "quadrature only; Monte Carlo rare events under-resolved at this budget",
        "nonclaims": ["NLL or sequence unbiasedness", "learning performance",
                      "paper replication", "TensorFlow/runtime admission"],
    }
    save(args.output / "contract.json", contract)

    def score(z, a, sigma, pi, n, eps):
        left, right = n // 2, n - n // 2
        logs = torch.stack((-.5 * ((z + a) / sigma).square(),
                            -.5 * ((z - a) / sigma).square()), -1)
        kernels = (logs - logs.amax(-1, keepdim=True)).exp()
        return pi * (1-pi) * (kernels[..., 1] - kernels[..., 0]) / (
            (1-pi + left*eps)*kernels[..., 0] + (pi + right*eps)*kernels[..., 1])

    def quadrature(order, a, sigma, pi, n, eps):
        # Jacobi matrix for probabilists' Hermite polynomials: standard normal.
        off = torch.arange(1, order, dtype=torch.float64).sqrt()
        nodes, eigenvectors = torch.linalg.eigh(torch.diag(off, 1) + torch.diag(off, -1))
        weights = eigenvectors[0].square()
        torch.testing.assert_close((weights*nodes).sum(), torch.tensor(0.), atol=1e-13, rtol=0)
        torch.testing.assert_close((weights*nodes.square()).sum(), torch.tensor(1.), atol=1e-13, rtol=0)
        z = torch.stack((-a + sigma*nodes, a + sigma*nodes))
        mass = torch.tensor([1-pi, pi])[:, None] * weights[None, :]
        s = score(z, a, sigma, pi, n, eps)
        ez, es, ezs = ((mass*v).sum().item() for v in (z, s, z*s))
        return {"E_z": ez, "E_score": es, "unnormalized": ezs,
                "normalized": (n-1)/(n*injected_constant) * (ezs-ez*es)}

    def predictions(a, sigma, pi, n):
        low = quadrature(64, a, sigma, pi, n, epsilon)
        high = quadrature(128, a, sigma, pi, n, epsilon)
        exact = 2*a*pi*(1-pi)
        for key in high:
            torch.testing.assert_close(torch.tensor(low[key]), torch.tensor(high[key]), atol=1e-13, rtol=1e-7)
        ideal = quadrature(128, a, sigma, pi, n, 0.)
        torch.testing.assert_close(torch.tensor(ideal["unnormalized"]), torch.tensor(exact), atol=1e-14, rtol=1e-7)
        torch.testing.assert_close(torch.tensor(ideal["E_score"]), torch.tensor(0.), atol=1e-13, rtol=0)
        return {"source": high, "lower_order": low, "ideal_derivative": exact,
                "normalized_ideal_factor": (n-1)/(n*injected_constant)}

    rows = []
    for case, a, sigma in (("overlap", .4, .8), ("separated", 3., .4)):
        for n in (2, 25):
            pi = .35
            pred = predictions(a, sigma, pi, n)
            draws, blocks = {"unnormalized": [], "normalized": []}, []
            max_algebra_error, max_constant_gradient = 0., 0.
            left = n // 2
            means = torch.zeros(batch, n, 3)
            means[:, :left, 0], means[:, left:, 0] = -a, a
            bandwidths = torch.tensor([sigma, .7, .9]).expand(batch, -1)
            for seed in seeds:
                torch.manual_seed(seed)
                phi = torch.full((batch,), math.log(pi/(1-pi)), requires_grad=True)
                p = phi.sigmoid()
                weights = torch.cat(((1-p[:, None]).expand(-1, left)/left,
                                     p[:, None].expand(-1, n-left)/(n-left)), 1)
                samples, injected, dynamics = model.resample_particles({
                    "particles_downscaled": means, "particle_weights": weights,
                    "bandwidths_downscaled": bandwidths, "do_resample": True})
                assert not samples.requires_grad
                torch.testing.assert_close(injected.detach(), torch.full_like(injected, injected_constant))
                measurement = torch.ones_like(injected)
                normalized = model.weigh_particles({}, injected, dynamics, lambda _: measurement, {})
                z = samples[..., 0]
                unnormalized_loss = (z * injected).mean(-1).sum()
                normalized_loss = (z * normalized).sum()
                actual_raw = torch.autograd.grad(unnormalized_loss, phi, retain_graph=True)[0]
                actual_norm = torch.autograd.grad(normalized_loss, phi, retain_graph=True)[0]
                constant = torch.autograd.grad(normalized.sum(), phi)[0]
                s = score(z, a, sigma, pi, n, epsilon)
                expected_raw = (z*s).mean(-1)
                expected_norm = ((z-z.mean(-1, keepdim=True))*s).mean(-1)/injected_constant
                for actual, expected in ((actual_raw, expected_raw), (actual_norm, expected_norm)):
                    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
                    assert torch.isfinite(actual).all()
                    max_algebra_error = max(max_algebra_error, (actual-expected).abs().max().item())
                torch.testing.assert_close(constant, torch.zeros_like(constant), atol=1e-12, rtol=0)
                max_constant_gradient = max(max_constant_gradient, constant.abs().max().item())
                draws["unnormalized"].append(actual_raw.detach())
                draws["normalized"].append(actual_norm.detach())
                blocks.append({"seed": seed, "unnormalized_mean": actual_raw.mean().item(),
                               "normalized_mean": actual_norm.mean().item()})
            comparisons = {}
            for key, tensors in draws.items():
                cloud = torch.cat(tensors)
                mean, variance = cloud.mean().item(), cloud.var(unbiased=True).item()
                mcse = math.sqrt(variance/cloud.numel())
                target = pred["source"][key]
                quad_delta = abs(target-pred["lower_order"][key])
                tolerance = 6*mcse + quad_delta + 1e-12
                assert abs(mean-target) <= tolerance, (case, n, key, mean, target, tolerance)
                lo, hi = mean-1.959963984540054*mcse, mean+1.959963984540054*mcse
                comparisons[key] = {"mean": mean, "variance": variance, "MCSE": mcse,
                    "approximate_95_interval": [lo, hi], "expected_source": target,
                    "six_MCSE_tolerance": tolerance, "clouds": cloud.numel(),
                    "ideal_derivative_in_interval": lo <= pred["ideal_derivative"] <= hi}
            rows.append({"case": case, "particles": n, "pi_right": pi,
                         "means_x": [-a, a], "sigma_x": sigma, "predictions": pred,
                         "comparisons": comparisons, "per_seed": blocks,
                         "maximum_algebra_error": max_algebra_error,
                         "maximum_constant_gradient": max_constant_gradient})

    rare = predictions(3., .4, 1e-10, 25)
    rare["sampling_status"] = "NOT_RUN_RARE_EVENT_MONTE_CARLO_UNDER_RESOLVED"
    rare["source_unnormalized_to_ideal_ratio"] = rare["source"]["unnormalized"]/rare["ideal_derivative"]
    rare["source_normalized_to_ideal_ratio"] = rare["source"]["normalized"]/rare["ideal_derivative"]
    save(args.output / "comparisons.json", {"ordinary": rows, "rare": rare})
    save(args.output / "result.json", {"check": "PASS", "cases": len(rows),
        "clouds": len(rows)*len(seeds)*batch,
        "source_algebra": "PASS", "source_expectation": "PASS",
        "source_vs_ideal": "DIFFERENT_EXPECTATIONS_DERIVED_AND_CHECKED",
        "whole_NLL_unbiasedness": "NOT_ESTABLISHED",
        "scientific_replication": "INCOMPLETE"})
    print(json.dumps({"check": "PASS", "cases": len(rows), "clouds": len(rows)*len(seeds)*batch}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--output")+1])
            out.mkdir(parents=True, exist_ok=True)
            save(out / "result.json", {"check": "FAILED", "error": repr(exc),
                                       "scientific_replication": "INCOMPLETE"})
        traceback.print_exc()
        raise
