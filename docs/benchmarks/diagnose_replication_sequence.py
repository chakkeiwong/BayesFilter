#!/usr/bin/env python3
"""CPU diagnostic: source truncated sequence gradient versus explicit densities.

Independent author PyTorch/NumPy reference only, not a BayesFilter runtime.
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
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "configuration", "training-artifact", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU diagnostic must hide GPU before import")
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("reference_training", Path(__file__).with_name("diagnose_replication_training.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    save = helper.save
    config = helper.restore_yaml_dimensions(json.loads(args.configuration.read_text()))
    src = helper.source_copy(args.author_root, args.output, torch)
    sys.path.insert(0, str(args.author_root / "packages/kernel-density-estimator-bandwdidth-prediction/packages"))
    sys.path.insert(0, str(src))
    from models.model_creator import create_model
    from models.kde_particle_filter.kde_particle_filter import KDEParticleFilter
    from kernel_density_estimation.kernel_density_estimator import KernelDensityEstimator
    from loss_functions.loss_functions import KDE_NLL, create_loss_function
    from trainers.specific_trainers.full_sequence_trainer import FullSequenceTrainer
    from problems.bearings_only.bearings_only_problem import BearingsOnlyObservationTransformer
    params = copy.deepcopy(next(e["parameters"] for e in config["experiments"]
        if e["name"] == "full_dpf_full_training_fixed_bands_000"))
    checkpoint = args.training_artifact / "full_training/models/full_dpf_model_best.pt"
    assert json.loads((args.training_artifact / "result.json").read_text())["check"] == "PASS"
    seeds = [924141, 924142, 924143]
    tol = {"atol": 3e-5, "rtol": 3e-4, "gradient_relative_l2": 3e-4}
    save(args.output / "contract.json", {"target": "Source truncated BPTT estimator at T=17,N=25,B=2",
        "tolerances": tol, "seeds": seeds, "model_seed": 924140,
        "checkpoint_sha256": helper.digest(checkpoint), "checkpoint": str(checkpoint),
        "cpu_only": True, "CUDA_VISIBLE_DEVICES": "-1",
        "trainer_constructor": "Not called; actual forward endpoint with explicit fixture fields",
        "nonclaims": ["untruncated total derivative", "unbiasedness", "training quality", "replication"]})
    def components(x, means, bandwidths, angle_epsilon):
        delta = x[:, :, None, :] - means[:, None, :, :]
        b = bandwidths[:, None, None, :]
        normal = (-.5 * (delta[..., :2] / b[..., :2]).square()
                  - b[..., :2].log() - .5 * math.log(2 * math.pi)).sum(-1)
        k = 1 / (b[..., 2] + angle_epsilon)
        return normal + k * delta[..., 2].cos() - math.log(2 * math.pi) - torch.special.i0e(k).log() - k.abs()
    def independent_log_prob(kde, x, do_normalize_weights=True):
        w = kde.weights + 1e-8
        if do_normalize_weights:
            w = w / w.sum(-1, keepdim=True)
        return torch.logsumexp(components(x, kde.particles, kde.bandwidths, 1e-8) + w[:, None, :].log(), -1)
    def independent_nll(loss, output, states):
        assert loss.manual_bandwidth is None
        density = (components(states[:, None, :], output["particles"], output["bandwidths"], 0.).exp()
                   * output["particle_weights"][:, None, :]).sum(-1)
        return -(density + 1e-8).log()
    t = torch.arange(17, dtype=torch.float32)
    x = (2. + .1 * t).repeat(2, 1)
    y = torch.tensor([.2, -.2]).unsqueeze(1).expand(2, 17)
    angle = torch.tensor([3.1, -3.1]).unsqueeze(1).expand(2, 17)
    data = {"states": torch.stack([x, y, angle], -1),
            "observations": torch.atan2(y, x+5.).unsqueeze(-1),
            "ground_truth_mask": (torch.arange(17) % 4 == 0).repeat(2, 1)}
    torch.manual_seed(924140)
    base = create_model(params["model_params"], config["architecture"]).train()
    assert base.differentiable_resampling_method == "ImportanceSampling"
    states_by_role = {"random_initialization": copy.deepcopy(base.state_dict())}
    base.load_pretrained({"dpf_model": str(checkpoint)}, "cpu")
    states_by_role["short_trained"] = copy.deepcopy(base.state_dict())
    rows, details = [], []
    original_resample = KDEParticleFilter.resample_particles
    original_nll = KDE_NLL.compute_loss
    original_logp = KernelDensityEstimator.log_prob
    for role, state_dict in states_by_role.items():
        for seed in seeds:
            runs = []
            for mode in ("source", "independent_density"):
                model = copy.deepcopy(base)
                model.load_state_dict(state_dict)
                model.zero_grad(set_to_none=True)
                model.train()
                trainer = FullSequenceTrainer.__new__(FullSequenceTrainer)
                trainer.main_model, trainer.device = model, "cpu"
                trainer.problem = SimpleNamespace(observation_transformer=BearingsOnlyObservationTransformer())
                trainer.number_of_particles = params["training_params"]["number_of_particles"]
                trainer.skip_first_step_for_loss = False
                trainer.max_sequence_length = None
                trainer.truncated_bptt_modulo = params["training_params"]["truncated_bptt_modulo"]
                trainer.loss_shaper = None
                trainer.loss_function = create_loss_function(params["training_params"]["loss_params"], model)
                assert trainer.truncated_bptt_modulo == 4
                samples, resampling_flags, calls = [], [], {"resample": 0, "nll": 0, "log_prob": 0}
                def resample(self, *a, **kw):
                    resampling_flags.append(bool(a[0]["do_resample"]))
                    result = original_resample(self, *a, **kw)
                    samples.append(result[0].detach().clone())
                    calls["resample"] += 1
                    return result
                def nll(self, *a, **kw):
                    calls["nll"] += 1
                    return (original_nll if mode == "source" else independent_nll)(self, *a, **kw)
                def logp(self, *a, **kw):
                    calls["log_prob"] += 1
                    return (original_logp if mode == "source" else independent_log_prob)(self, *a, **kw)
                torch.manual_seed(seed)
                with patch.object(KDEParticleFilter, "resample_particles", resample), \
                     patch.object(KDE_NLL, "compute_loss", nll), \
                     patch.object(KernelDensityEstimator, "log_prob", logp):
                    loss, batch = trainer.do_forward_pass(copy.deepcopy(data), None, 0)
                    assert batch == 2 and torch.isfinite(loss).item()
                    loss.backward()
                assert calls == {"resample": 16, "nll": 16, "log_prob": 15}, calls
                assert resampling_flags == [False] + [True] * 15, resampling_flags
                gradients = {name: None if p.grad is None else p.grad.detach().clone()
                             for name, p in model.named_parameters()}
                runs.append({"loss": loss.detach(), "gradients": gradients, "samples": samples, "calls": calls, "resampling_flags": resampling_flags})
            left, right = runs
            torch.testing.assert_close(left["loss"], right["loss"], atol=tol["atol"], rtol=tol["rtol"])
            assert all(torch.equal(a, b) for a, b in zip(left["samples"], right["samples"]))
            total_sq, error_sq, count, absent = 0., 0., 0, []
            for name, g in left["gradients"].items():
                h = right["gradients"][name]
                assert (g is None) == (h is None), name
                if g is None:
                    absent.append(name)
                    continue
                assert torch.isfinite(g).all().item() and torch.isfinite(h).all().item(), name
                torch.testing.assert_close(g, h, atol=tol["atol"], rtol=tol["rtol"], msg=name)
                norm = float(g.double().square().sum())
                error = float((g.double()-h.double()).square().sum())
                total_sq += norm
                error_sq += error
                count += 1
                details.append({"role": role, "seed": seed, "parameter": name,
                    "l2_norm": math.sqrt(norm), "error_l2": math.sqrt(error),
                    "max_absolute_error": float((g-h).abs().max())})
            assert total_sq > 1e-12, "Degenerate fixture gradient signal"
            relative_error = math.sqrt(error_sq / total_sq)
            assert relative_error <= tol["gradient_relative_l2"], relative_error
            rows.append({"role": role, "seed": seed, "source_loss": float(left["loss"]),
                "independent_loss": float(right["loss"]), "gradient_relative_l2": relative_error,
                "gradient_l2": math.sqrt(total_sq), "parameters_with_gradients": count,
                "parameters_without_gradients": absent, "calls_per_arm": left["calls"], "resampling_flags": left["resampling_flags"], "samples_equal": True})
            save(args.output / "comparisons.json", {"cases": rows, "parameter_errors": details})
    save(args.output / "result.json", {"check": "PASS", "cases": len(rows), "cpu_only": True,
        "sequence_truncated_parameter_gradient_parity": "PASS_ON_DIAGNOSTIC_FIXTURES",
        "max_gradient_relative_l2": max(row["gradient_relative_l2"] for row in rows),
        "unbiasedness": "NOT_ESTABLISHED", "untruncated_total_gradient": "NOT_CHECKED",
        "scientific_replication": "INCOMPLETE", "optimizer_updates": 0,
        "backend_role": "Independent author PyTorch/NumPy reference; CPU eager"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        output = Path(sys.argv[sys.argv.index("--output")+1])
        (output / "error.txt").write_text(traceback.format_exc())
        (output / "result.json").write_text(json.dumps({"check": "FAILED", "detail": repr(exc)}) + "\n")
        raise
