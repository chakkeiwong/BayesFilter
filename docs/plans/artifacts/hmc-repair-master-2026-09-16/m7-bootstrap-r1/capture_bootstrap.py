"""Diagnostic extraction comparison; scripted control flow, no posterior claim."""
import dataclasses
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from bayesfilter.inference import hmc_kernel_tuning as legacy
from tests.test_hmc_kernel_tuning_bootstrap import (
    _ToyGaussianAdapter, _MismatchedAdapter, _config, _geometry, _fake_result,
)

cases = {
    "pass": ([.70], {}, {}),
    "low_repair": ([.40, .70], {}, {}),
    "high_repair": ([.95, .70], {}, {}),
    "bracket_repair": ([.40, .95, .70], {}, {}),
    "repair_exhaustion": ([.95, .95], {"max_repairs": 1}, {}),
    "invalid_log_accept": ([.70], {}, {"finite_log_accept": False}),
    "invalid_state": ([.70], {}, {"finite_samples": False}),
    "invalid_target": ([.70], {}, {"target_log_prob_finite": False}),
    "runtime_failure": ([.70], {}, {"runtime_s": float("nan")}),
}
records = {}
for name, (acceptances, settings, fake_options) in cases.items():
    calls = []
    def runner(adapter, state, config):
        calls.append({"step": config.step_size, "L": config.num_leapfrog_steps,
                      "seed": config.seed, "adapter": adapter.adapter_signature(),
                      "initial_state": state.numpy().tolist()})
        return _fake_result(acceptance=acceptances.pop(0), **fake_options)
    result = legacy.run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(), geometry=_geometry(),
        config=_config(**settings), run_full_chain=runner,
    )
    records[name] = {"payload": result.payload(), "hash": result.artifact_hash,
                     "calls": calls}

for name, invoke in {
    "adapter_mismatch": lambda: legacy.run_hmc_bootstrap_screen(
        adapter=_MismatchedAdapter(), geometry=_geometry(), config=_config()),
    "invalid_repair_factor": lambda: _config(step_repair_factor=1.),
    "invalid_seed": lambda: _config(seed=(1,)),
}.items():
    try:
        invoke()
    except Exception as exc:
        records[name] = {"exception": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError(f"{name} did not reject invalid inputs")

geometry = _geometry(initial_covariance=[[2., .4], [.4, .8]],
                     initial_position=[.2, -.3])
adapter = legacy._build_bootstrap_fixed_mass_adapter(
    adapter=_ToyGaussianAdapter(), mass_artifact=geometry.mass_artifact,
    mass_signature=geometry.mass_artifact_signature,
    target_scope="kernel_bootstrap_toy_gaussian",
)
for name, coordinates in {
    "scalar_transform": [.2, -.5],
    "chain_transform": [[.2, -.5], [.4, .1]],
    "draw_chain_transform": [[[.2, -.5], [.4, .1]], [[.8, -.3], [.2, .9]]],
}.items():
    values, scores = adapter.log_prob_and_grad(coordinates)
    records[name] = {"signature": adapter.adapter_signature(),
                     "positions": adapter.latent_to_position(coordinates).numpy().tolist(),
                     "values": values.numpy().tolist(), "scores": scores.numpy().tolist(),
                     "capability": dataclasses.asdict(adapter.value_score_capability())}

def diagnostic_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    return value


records = diagnostic_json(records)
path = Path(sys.argv[1])
path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
if len(sys.argv) > 2:
    assert json.loads(path.read_text()) == json.loads(Path(sys.argv[2]).read_text()), (
        "bootstrap payloads, hashes, repair/seed sequences, errors or transforms changed"
    )
print(json.dumps({"cases": len(records), "exact_parity": len(sys.argv) > 2}))
