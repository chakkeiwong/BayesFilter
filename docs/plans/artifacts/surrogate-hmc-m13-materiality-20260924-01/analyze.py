"""Diagnostic-only postprocessing of existing M13 likelihood results."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

start, cpu = time.monotonic(), time.process_time()
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "docs/plans/artifacts"
paths = {
    "reference": BASE / "surrogate-hmc-closure-20260922-m13-reference-03/m13-reference-trajectory/report.json",
    "saved": BASE / "surrogate-hmc-closure-20260922-m13-consumer-03/m13-consumer-localize/report.json",
    "replay": BASE / "surrogate-hmc-closure-20260922-m13-repair-01/m13-consumer-repair/report.json",
}
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
data = {name: json.loads(path.read_text()) for name, path in paths.items()}
ref, saved, replay = data["reference"], data["saved"], data["replay"]
assert ref["likelihood_reference_qualified"] and ref["replicates"] == 16
assert saved["reference_sha256"] == replay["reference_sha256"] == hashes["reference"]
assert replay["previous_localization_sha256"] == hashes["saved"]
arm = next(a for a in saved["arms"] if a["particle_count"] == 256 and a["dtype"] == "float32")
assert len(ref["numerical_rows"]) == len(arm["rows"]) == len(replay["fp32_replay"]["value"]) == 9
rows = []
for i, (r, old, value) in enumerate(zip(ref["numerical_rows"], arm["rows"], replay["fp32_replay"]["value"])):
    assert r["theta"] == old["theta"] == replay["parameter_points"][i]
    assert old["program_valid"] and r["qualified"]
    f = r["particle_levels"]["65536"]["full"]
    estimates = f["log_estimates"]
    assert len(estimates) == 16 and all(math.isfinite(x) for x in estimates)
    shift = max(estimates)
    weights = [math.exp(x-shift) for x in estimates]
    mean = statistics.mean(weights)
    se = statistics.stdev(weights) / math.sqrt(len(weights))
    ell, log_se = shift+math.log(mean), se/mean
    assert abs(ell-f["log_mean"]) < 1e-11
    assert abs(log_se-f["coefficient_of_variation"]/4) < 1e-12
    rows.append(dict(index=i, theta=r["theta"],
        noise_sd_multipliers=[math.exp(x) for x in r["theta"]],
        reference_log_mean=ell, reference_log_mcse_delta_method=log_se,
        reference_likelihood_mean=math.exp(ell),
        reference_likelihood_mcse=math.exp(ell)*log_se,
        reference_log_interval=[f["log_lower"], f["log_upper"]],
        reference_log_ci_halfwidth=f["log_halfwidth"],
        saved_log_likelihood=old["log_likelihood"], replay_log_likelihood=value,
        replay_shift=value-old["log_likelihood"], replay_minus_reference=value-ell))
deltas = [r["replay_shift"] for r in rows]
top = sorted(rows, key=lambda r: r["reference_log_mean"], reverse=True)[:2]
a, b = top
contrast_delta = a["replay_shift"]-b["replay_shift"]
ref_contrast = a["reference_log_mean"]-b["reference_log_mean"]
replay_contrast = a["replay_log_likelihood"]-b["replay_log_likelihood"]
for r in rows:
    r["absolute_replay_shift_in_reference_mcse"] = abs(r["replay_shift"])/r["reference_log_mcse_delta_method"]
    r["reference_likelihood_relative_to_grid_maximum"] = math.exp(r["reference_log_mean"]-a["reference_log_mean"])
result = dict(quantity="saved_result_materiality_diagnostics",
    posterior_qualified=False, dctr_randomized_design_mcse=None,
    reference_replicates=16, reference_particles_per_replicate=65536,
    reference_mcse_method="sample SD of likelihood estimates / sqrt(16); delta method for log of mean",
    rows=rows, full_grid=dict(maximum_absolute_replay_shift=max(map(abs,deltas)),
        maximum_absolute_pair_log_ratio_change=max(deltas)-min(deltas),
        maximum_pair_ratio_factor=math.exp(max(deltas)-min(deltas))),
    highest_reference_grid_pair=dict(indices=[a["index"],b["index"]],
        log_ratio_replay_change=contrast_delta, ratio_replay_change_factor=math.exp(contrast_delta),
        reference_log_ratio=ref_contrast,
        reference_log_ratio_mcse=math.hypot(a["reference_log_mcse_delta_method"],b["reference_log_mcse_delta_method"]),
        replay_log_ratio=replay_contrast,
        replay_ratio_over_reference_ratio=math.exp(replay_contrast-ref_contrast)),
    checks=dict(reference_hashes_match=True, saved_hash_matches=True, all_nine_points_match=True,
        recomputed_reference_mean_and_cv_match=True),
    nonclaims=["No new HMC result or posterior materiality decision",
        "Reference MCSE does not estimate fixed-design DCTR bias or variability",
        "Highest-likelihood grid pair is post-hoc illustration, not posterior support",
        "Likelihood-ratio changes apply to potential shape; they are not measured HMC acceptance changes"],
    input_sha256=hashes)
(OUT/"report.json").write_text(json.dumps(result, indent=2)+"\n")
manifest = dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    command="/home/chakwong/anaconda3/envs/tftwogpu/bin/python "+str(Path(__file__).relative_to(ROOT)),
    environment="tftwogpu interpreter; Python standard library only; no TensorFlow import",
    cpu_gpu_status="CPU-only JSON postprocessing; no GPU detected or used",
    data_version={name:dict(path=str(paths[name].relative_to(ROOT)),sha256=hashes[name]) for name in paths},
    random_seeds="No new random draws; reference seed schedule and frozen design are in hashed input reports",
    plan_file="docs/plans/surrogate-hmc-m13-materiality-2026-09-24.md",
    result_file=str((OUT/"report.json").relative_to(ROOT)),
    cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)
(OUT/"run-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(checks=result["checks"],log_mcse_range=[min(r["reference_log_mcse_delta_method"] for r in rows),max(r["reference_log_mcse_delta_method"] for r in rows)],
    full_grid=result["full_grid"],highest_reference_grid_pair=result["highest_reference_grid_pair"],
    cpu_seconds=manifest["cpu_seconds"],wall_seconds=manifest["wall_seconds"],output=str(OUT.relative_to(ROOT))),indent=2))
