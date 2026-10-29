"""M13-scoped energy repair evidence; no independent tuning authority."""
import json
from pathlib import Path
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash

STEP_CAP = .949153975512196 / 2.
PROBE = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-development-01/energy-step-probe-03/report.json"
HEALTH_KEYS = ("finite","endpoint_valid","all_chains_moved","energy_valid","no_divergences","no_force_fallback")


def checked_energy_repair_probe(binding, config, *, path=None):
    """Require completed GPU evidence for this actual binding and step ceiling."""
    path = PROBE if path is None else Path(path)
    probe = json.loads(path.read_text())
    receipt = json.loads(path.with_name("guard-receipt.json").read_text())
    reps = probe.get("replications", [])
    if (probe.get("status") != "DIAGNOSTIC_COMPLETE"
            or probe.get("repair_screen_passed") is not True
            or probe.get("source_unchanged") is not True
            or receipt.get("status") != "COMPLETED"
            or receipt.get("accounting_complete") is not True
            or probe.get("device") != "gpu" or probe.get("jit_compile") is not True
            or probe.get("counts_per_chain") != 64
            or probe.get("multipliers") != [.125, 1.]
            or probe.get("step_cap") != STEP_CAP
            or probe.get("original_step") != 2*STEP_CAP
            or len(reps) != 2
            or any(rep.get("health_passed") is not True or
                   any(rep.get("checks", {}).get(k) is not True for k in HEALTH_KEYS)
                   for rep in reps)):
        raise ValueError("energy repair requires the completed two-start GPU health screen")
    if probe.get("binding") != json.loads(json.dumps(binding.payload())):
        raise ValueError("energy repair probe binding mismatch")
    if (config.nomination.max_step_size != STEP_CAP
            or config.nomination.energy_error_max != config.energy_error_max
            or config.energy_error_max != 1000.):
        raise ValueError("energy repair config differs from the screened scope")
    return dict(passed=True, probe_path=str(path), probe_sha256=file_hash(path),
        receipt_sha256=file_hash(path.with_name("guard-receipt.json")),
        step_cap=STEP_CAP, energy_error_max=1000.,
        role="discarded health screen only; no convergence or posterior claim")
