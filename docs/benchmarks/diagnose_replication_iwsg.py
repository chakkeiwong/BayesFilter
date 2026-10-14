#!/usr/bin/env python3
"""Independent CPU author-reference diagnostic, not a BayesFilter runtime.

Fixed-sample source resampling -> normalization -> KDE NLL derivative checks.
NumPy/PyTorch in imported author code retain their independent-reference role.
"""
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
from unittest.mock import patch


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU-only reference must hide GPUs before framework import")
    import torch
    spec = importlib.util.spec_from_file_location("reference_training", Path(__file__).with_name("diagnose_replication_training.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config = helper.restore_yaml_dimensions(json.loads(args.configuration.read_text()))
    src = helper.source_copy(args.author_root, args.output, torch)
    sys.path.insert(0, str(args.author_root / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
    sys.path.insert(0, str(src))
    from models.model_creator import create_model
    from kernel_density_estimation.kernel_density_estimator import KernelDensityEstimator
    from loss_functions.loss_functions import create_loss_function
    params = copy.deepcopy(next(e["parameters"] for e in config["experiments"]
                                if e["name"] == "full_dpf_full_training_fixed_bands_000"))
    torch.manual_seed(924101)
    model = create_model(params["model_params"], config["architecture"]).train()
    assert model.differentiable_resampling_method == "ImportanceSampling"
    assert not model.decouple_weights_for_resampling and not model.decouple_bandwidths_for_resampling
    loss_params = params["training_params"]["loss_params"]
    loss_fn = create_loss_function(loss_params, model)
    assert type(loss_fn).__name__ == "KDE_NLL" and loss_fn.manual_bandwidth is None
    assert [v["distribution_type"] for v in model.kde_params["dims"].values()] == ["Normal", "Normal", "Von_Mises"]

    # This expression does not call the source KDE or distribution log_prob.
    def components(x, means, bandwidths, angle_epsilon):
        delta = x[:, :, None, :] - means[:, None, :, :]
        b = bandwidths[:, None, None, :]
        normal = (-.5 * (delta[..., :2] / b[..., :2]).square()
                  - b[..., :2].log() - .5 * math.log(2 * math.pi)).sum(-1)
        concentration = 1 / (b[..., 2] + angle_epsilon)
        log_i0 = torch.special.i0e(concentration).log() + concentration.abs()
        return normal + concentration * delta[..., 2].cos() - math.log(2 * math.pi) - log_i0

    def mixture(x, means, weights, bandwidths, epsilon):
        w = weights + epsilon
        w = w / w.sum(-1, keepdim=True)
        return torch.logsumexp(components(x, means, bandwidths, 1e-8) + w[:, None, :].log(), -1)

    tolerances = {"float64": {"atol": 3e-7, "rtol": 3e-6},
                  "float32": {"atol": 3e-5, "rtol": 3e-4},
                  "finite_difference": {"step": 1e-5, "atol": 3e-7, "rtol": 3e-5}}
    save(args.output / "contract.json", {"tolerances": tolerances,
         "reason": "Float64 allows source VonMises polynomial approximation; FP32 allows rounding through exp/log and normalization.",
         "seeds": [924101], "sampling": "fixed samples; denominator held fixed at base parameter",
         "nonclaims": ["Monte Carlo unbiasedness", "full sequence parameter gradients", "learning quality", "paper replication"]})
    rows, mismatches = [], []
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)  # Source scratch buffers otherwise silently default to FP32.
        model.to(dtype=dtype)
        tol = tolerances[str(dtype).split(".")[-1]]
        for case in ("overlap", "separated", "tiny_weight_wrap", "single_component"):
            means = torch.tensor([[[-.4, .2, math.pi-.03], [.5, -.3, -math.pi+.04], [0., .4, .15]]])
            if case == "separated":
                means[0, :, 0] *= 8
            raw_weights = torch.tensor([[.2, .3, .5]])
            if case == "tiny_weight_wrap":
                raw_weights = torch.tensor([[1e-14, .4, .6]])
            if case == "single_component":
                means, raw_weights = means[:, :1], torch.ones((1, 1))
            bandwidths = torch.tensor([[.65, .8, .3]])
            x = means.detach().clone()
            x[..., 0] += .17
            x[..., 1] -= .11
            x[..., 2] += .09
            # Use the source angular normalization for the fixed observation locations.
            x = model.particle_transformer.apply_norm(x).detach()
            target = torch.tensor([[.1, -.2, -math.pi+.07]])
            evaluation_bandwidth = torch.tensor([[.75, .9, .4]])
            logits = raw_weights.log().requires_grad_()
            means = means.requires_grad_()
            log_b = bandwidths.log().requires_grad_()
            variables = (means, logits, log_b)
            weights, bandwidths = logits.softmax(-1), log_b.exp()
            base_log = mixture(x, means, weights, bandwidths, 1e-8).detach()
            input_dict = {"particles_downscaled": means, "particle_weights": weights,
                          "bandwidths_downscaled": bandwidths, "do_resample": True}
            with patch.object(KernelDensityEstimator, "sample", return_value=x.clone()) as sampler:
                samples, injected, dynamics = model.resample_particles(input_dict)
            assert sampler.call_count == 1 and not samples.requires_grad
            measurement = torch.linspace(.7, 1.3, means.shape[1]).reshape(1, -1)
            normalized = model.weigh_particles({}, injected, dynamics, lambda _: measurement, {})
            source_loss = loss_fn.compute_loss({"particles": samples, "particle_weights": normalized,
                                               "bandwidths": evaluation_bandwidth}, target).sum()
            source_grad = torch.autograd.grad(source_loss, variables, retain_graph=True)

            def reference(vs, denominator=base_log):
                m, l, lb = vs
                logp = mixture(x, m, l.softmax(-1), lb.exp(), 1e-8)
                iw = torch.exp(logp - denominator) + 1e-8
                nw = measurement * iw
                nw = nw / nw.abs().sum(-1, keepdim=True).clamp_min(1e-8)
                density = (components(target[:, None, :], x, evaluation_bandwidth, 0.).exp()
                           * nw[:, None, :]).sum(-1)
                return -(density + 1e-8).log().sum()

            reference_loss = reference(variables)
            reference_grad = torch.autograd.grad(reference_loss, variables, retain_graph=True)
            torch.testing.assert_close(source_loss, reference_loss, **tol)
            for actual, expected in zip(source_grad, reference_grad):
                torch.testing.assert_close(actual, expected, **tol)
            kde = KernelDensityEstimator(model.kde_params, means, weights, bandwidths)
            source_log = kde.log_prob(x)
            independent_log = mixture(x, means, weights, bandwidths, 1e-8)
            torch.testing.assert_close(source_log, independent_log, **tol)
            source_score = torch.autograd.grad(source_log.sum(), variables, retain_graph=True)
            independent_score = torch.autograd.grad(independent_log.sum(), variables, retain_graph=True)
            for actual, expected in zip(source_score, independent_score):
                torch.testing.assert_close(actual, expected, **tol)
            torch.testing.assert_close(injected, torch.ones_like(injected) + 1e-8, **tol)
            torch.testing.assert_close(normalized.sum(-1), torch.ones(1), **tol)
            wrapped = x.clone()
            wrapped[..., 2] += 2 * math.pi
            torch.testing.assert_close(kde.log_prob(wrapped), source_log, **tol)
            # Constant downstream loss must have zero upstream derivative.
            zero = torch.autograd.grad(normalized.sum(), variables, retain_graph=True)
            assert max(float(g.abs().max()) for g in zero) < tol["atol"]
            fd_error = None
            if dtype == torch.float64:
                direction = tuple(torch.linspace(-.5, .6, v.numel()).reshape(v.shape) for v in variables)
                h = tolerances["finite_difference"]["step"]
                finite_diff = (reference(tuple(v.detach()+h*d for v,d in zip(variables,direction)))
                               - reference(tuple(v.detach()-h*d for v,d in zip(variables,direction)))) / (2*h)
                analytical = sum((g*d).sum() for g,d in zip(source_grad,direction))
                torch.testing.assert_close(analytical, finite_diff, atol=3e-7, rtol=3e-5)
                fd_error = float((analytical-finite_diff).abs())
            # Source sampler uses raw normalized weights; log_prob uses smoothed weights.
            smoothed = (weights.detach()+1e-8) / (weights.detach()+1e-8).sum(-1, keepdim=True)
            mismatch = float((smoothed-weights.detach()).abs().max())
            raw_log = mixture(x, means, weights, bandwidths, 0.)
            ideal_score = torch.autograd.grad(raw_log.sum(), variables)
            score_delta = max(float((a-b).abs().max()) for a,b in zip(source_score,ideal_score))
            rows.append({"dtype": str(dtype), "case": case, "check": "PASS",
                         "loss_abs_error": float((source_loss-reference_loss).detach().abs()),
                         "gradient_max_abs_error": max(float((a-b).abs().max()) for a,b in zip(source_grad,reference_grad)),
                         "directional_fd_abs_error": fd_error,
                         "mixture_score_max_abs_error": max(float((a-b).abs().max()) for a,b in zip(source_score,independent_score)),
                         "source_vs_unsmoothed_weight_score_max_abs_delta": score_delta})
            mismatches.append({"case": case, "dtype": str(dtype), "max_component_probability_delta": mismatch})
    save(args.output / "comparisons.json", rows)
    save(args.output / "source-target-differences.json", {"sampling_vs_density_weights": mismatches,
        "source_sampling": "Categorical(raw weights normalized)",
        "source_resampling_density": "Normalize(weights+1e-8); angle concentration 1/(bandwidth+1e-8)",
        "source_injected_weights": "exp(log_density-stop_gradient(log_density))+1e-8",
        "source_nll": "-log(sum(normalized next weights * Gaussian/vonMises density)+1e-8); angle concentration 1/bandwidth",
        "verdict": "Source is not exactly the unsmoothed matched-proposal estimator; no blanket unbiasedness claim."})
    save(args.output / "result.json", {"check": "PASS", "cases": len(rows),
        "resampling_normalization_nll_fixed_sample_parity": "PASS", "cpu_only": True,
        "float64_source_scratch_buffers": "torch default dtype explicitly changed for independent reference check",
        "backend_role": "Independent author PyTorch reference; CPU intentionally; not TF/XLA",
        "source_paper_identity": "DIFFERS_BY_DOCUMENTED_EPSILONS",
        "sequence_parameter_gradient_parity": "NOT_CHECKED", "unbiasedness": "NOT_ESTABLISHED",
        "scientific_replication": "INCOMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--output")+1])
            (out / "diagnostic-error.log").write_text(traceback.format_exc())
            save(out / "result.json", {"check": "FAILED", "error": repr(exc), "scientific_replication": "INCOMPLETE"})
        raise
