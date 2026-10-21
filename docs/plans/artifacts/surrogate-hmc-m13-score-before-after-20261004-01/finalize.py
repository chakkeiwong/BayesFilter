"""Assemble the measured result and provenance; diagnostic/reporting only."""
import datetime, hashlib, json, math, pathlib, subprocess
R=pathlib.Path(__file__).resolve().parent
ROOT=R.parents[3]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
report=read(R/"report.json")
s=report["summary"]
common=read(R/"inputs.json")
scope=read(R/"saved-chain-scope-check.json")
sources=read(R/"source-snapshots.json")
manifest=read(R/"run-manifest.json")
assert all(scope["matched_fields"].values()) and not scope["saved_force_source_mismatches"]
assert s["all_points_valid"] and s["all_fd_checks_stable"]
assert s["maximum_endpoint_value_change"]==s["maximum_endpoint_score_component_change"]==0
assert all(s["fp32_repeatability"][a]["all_eight_runtime_force_parity_exact"] for a in ("before","after"))
assert all(s["fp32_repeatability"][a]["all_eight_runtime_force_valid"] for a in ("before","after"))
workers=[]
for arm,dtype in (("before","float32"),("after","float32"),("before","float64"),("after","float64")):
    path=(R if (arm,dtype)==("before","float32") else R/"attempt-02")/(arm+"-"+dtype+".json")
    d=read(path)
    assert d["status"]=="COMPLETED"
    assert d["input_manifest_sha256"]==sha(R/"inputs.json")
    assert d["source_snapshot_sha256"]==sha(R/"source-snapshots.json")
    for rel,h in d["imported_source_hashes"].items():
        assert sources["sources"][arm][rel]==h==sha(R/("source_"+arm)/rel)
    assert d["environment"]["gpu_memory_policy"]["all_physical_devices_memory_growth"]
    assert d["environment"]["jit_compile"]
    assert not d["environment"]["gpu_intentionally_hidden"]
    workers.append((path,d))
observed={a:read(R/"observability-01"/(a+".json")) for a in ("before","after")}
for d in observed.values():
    for b in d["base"]:
        for key in ("program_valid","value_valid","force_valid"):
            assert all(b["status"][key])
        assert all(x==0 for x in b["status"]["force_failure_code"]+b["status"]["value_failure_code"])
    assert all(e["valid"] and all(e["diagnostics"]["valid"]) for e in d["endpoint"])
ess=[v for d in observed.values() for e in d["endpoint"] for v in e["diagnostics"]["minimum_ess"]]
weight=[v for d in observed.values() for e in d["endpoint"] for v in e["diagnostics"]["proposal_maximum_normalized_weight"]]
obmanifest=read(R/"observability-01"/"manifest.json")
base_workers=[w for w in manifest["workers"] if "--worker" in w["command"]]
total=sum(w["wall_seconds"] for w in base_workers)+sum(w["wall_seconds"] for w in obmanifest["workers"])+common["wall_seconds"]
assert total<3600 and all(w["exit_code"]==0 for w in obmanifest["workers"])
assert len(base_workers)==5 and sum(w["exit_code"]!=0 for w in base_workers)==1
rows=report["probes"]
table=["| Fixed visited-state probe | LEDH score, before FP64 | LEDH score, after FP64 | Acceptance-likelihood score | LEDH minus acceptance log likelihood |",
       "|---|---:|---:|---:|---:|"]
vec=lambda x:"("+", ".join(f"{v:.5f}" for v in x)+")"
for r in rows:
    b,a=r["arms"]["before"],r["arms"]["after"]
    table.append(f'| {r["probe"]["label"]} | {vec(b["score_fp64"])} | {vec(a["score_fp64"])} | {vec(a["endpoint_likelihood_score"])} | {a["likelihood_value_fp64"]-a["endpoint_value"]:.5f} |')
heur=["| Probe | Updated LEDH | Zero | Prior only | Saved Gaussian |",
      "|---|---:|---:|---:|---:|"]
for r in rows:
    vals=[r["arms"]["after"]["runtime_posterior_score_vs_endpoint"]["whitened_l2_error"]]+[r["heuristics"][k]["error"]["whitened_l2_error"] for k in ("zero","prior_only","saved_gaussian")]
    heur.append("| "+r["probe"]["label"]+" | "+" | ".join(f"{v:.5f}" for v in vals)+" |")
curves=["| Central-difference step | Before: maximum component error | After: maximum component error |","|---|---:|---:|"]
for h in ("0.001","0.0003","0.0001"):
    curves.append(f'| {h} | {s["finite_difference_curves"]["before"][h]:.8g} | {s["finite_difference_curves"]["after"][h]:.8g} |')
first=rows[0]
note=f"""# M13 score measurement before and after main synchronization — 2026-10-04

The merge produces **no material improvement in M13 score accuracy at these eight
fixed probes**. Both versions agree with the numerical derivative of their own
finite LEDH likelihood. The large remaining discrepancy is between that
likelihood's score and the score of the separate likelihood used for HMC
acceptance. At the low-Q probes and the pooled mean, even the zero, prior-only,
and saved-Gaussian scores are closer to the acceptance score than the LEDH score.

This completes the requested measurement within the surrogate-HMC test campaign.
It does not complete the all-model campaign or resolve M13's mixing problem.
No HMC transitions, fresh tuning, or runtime/default changes were made.

## What was compared

M13 is the range-bearing LG state / nonlinear observation model: state dimension
4, observation dimension 2, horizon 20, data seed 1704. Parameters are
theta=(lambda_Q, lambda_R), with Q=Q0 exp(2 lambda_Q) and R=R0 exp(2 lambda_R).
The actual frozen proposal uses the analytical canonical LEDH score with 4096
particles. HMC acceptance uses the frozen defensive trajectory-importance
likelihood (8 replicates, 4096 trajectories per replicate), with its existing
seed [2026092402, 9101]. Both posterior scores include the same Gaussian prior,
with standard deviation log(2).

For score g=d log likelihood/d theta, the proposal potential gradient is
-g+theta/log(2)^2. Under theta=center+L z, the posterior score becomes L^T g_post.
The code evaluates the actual runtime force wrapper and verifies this relation
at all eight probes in both source versions. The FD reference is a derivative
of the **finite acceptance likelihood**, not an exact integrated model score.

The pre-sync snapshot reconstructs the saved working source from stash
2ca7b4f06b623cdfe694ed67c2fde89449cecf37 plus hash-matched original untracked
files. The post-sync snapshot records HEAD a55c6be7a16e9401242ee3449248fa347eb58d90
plus its local modifications. All saved-run force dependency hashes and all
scope fields matched. Main's KSC observation callback is not called by M13:
M13 uses make_range_bearing_model and polar_observation callbacks. Seven shared
dependencies changed, so the numerical comparison was still necessary.

The points were frozen before score evaluation: archived rows at process-scale
percentiles 1, 5, 25, 50, 75, 95, 99 and the pooled mean. These are states visited
by an incompletely mixed chain, not eight independent posterior draws. Exact
coordinates, source hashes, original row indices, data, controls and designs
are in [inputs.json](inputs.json) and [source-snapshots.json](source-snapshots.json).
The [plan](../../surrogate-hmc-m13-score-before-after-2026-10-04.md) records
assumptions, comparison rules, limits, and the pre-run skeptical audit.

## Derivative accuracy and precision

The largest pre/post change in an FP64 analytical score component is
{s["maximum_fp64_score_component_change"]:.5g}. The largest component discrepancy
between analytical score and its own finest central difference is
{s["before"]["maximum_own_derivative_component_error"]:.5g} before and
{s["after"]["maximum_own_derivative_component_error"]:.5g} after. Errors decrease
with the expected central-difference refinement:

{chr(10).join(curves)}

Every base value, derivative evaluation, endpoint evaluation, and finite-
difference stability check passed. The evidence supports derivative agreement
at these points and this fixed design; it is not a proof over the whole domain.
FP64 promotes the once-rounded FP32 inputs and coefficients and approximates the
smooth finite algorithm, not differentiation of floating-point rounding.

In the primary FP32 runs, maximum relative score error versus the corresponding
FP64 score was {100*s["before"]["maximum_fp32_score_relative_error_to_fp64"]:.3f}%
before and {100*s["after"]["maximum_fp32_score_relative_error_to_fp64"]:.3f}% after.
Cross-process repeat evaluations varied by at most
{s["fp32_repeatability"]["before"]["maximum_repeat_score_component_change"]:.6f}
and {s["fp32_repeatability"]["after"]["maximum_repeat_score_component_change"]:.6f}
in a component, or {100*s["fp32_repeatability"]["before"]["maximum_repeat_score_relative_change"]:.3f}%
and {100*s["fp32_repeatability"]["after"]["maximum_repeat_score_relative_change"]:.3f}%
in relative vector norm. Repeat log-likelihood differences were at most
{s["fp32_repeatability"]["before"]["maximum_repeat_value_change"]:.6f} and
{s["fp32_repeatability"]["after"]["maximum_repeat_value_change"]:.6f}.
This is observed execution variation across processes; its compiler/kernel
cause was not isolated, and it does not establish within-process randomness.

Consequently the report's primary-run FP32 pointwise improvement/worsening
labels describe those evaluations only. Their FD error allowance does not
include this execution variation. They cannot establish a reproducible merge
benefit or a statistical ranking. These small variations do not explain the
much larger acceptance-score discrepancy or justify changing the FP32 default.

## Agreement with the acceptance likelihood

Components below are likelihood derivatives with respect to (lambda_Q,lambda_R),
before adding the common prior. FP64 is shown to separate approximation error
from FP32 execution variation.

{chr(10).join(table)}

At the first point theta={vec(first["probe"]["theta"])}, the **actual FP32**
likelihood score changes from {vec(first["arms"]["before"]["score_fp32"])} to
{vec(first["arms"]["after"]["score_fp32"])}. The acceptance-likelihood score is
{vec(first["arms"]["after"]["endpoint_likelihood_score"])}. Its componentwise FD
refinement discrepancies are 2.14e-7 and 1.01e-6. The roughly 20-fold excess in
the Q score and 29-fold excess in the R score is far beyond reference resolution
or FP32 variation.

Acceptance-likelihood values and derivative estimates are bitwise identical
between the two source versions. The LEDH-minus-acceptance log-likelihood
difference ranges from about -34.88 to -0.34 over these points; this is not a
constant offset that disappears on differentiation.

The minimum trajectory-importance ESS ranges from {min(ess):.2f} to {max(ess):.2f}
out of 4096, and the largest normalized trajectory weight is {max(weight):.6f}.
These are integration diagnostics, not MCMC ESS or proof that this finite
likelihood equals the model's integrated likelihood.

## Constructed simple score comparisons

The entries are whitened posterior-score L2 error against the same finite
acceptance target. Zero ignores all curvature, prior-only ignores the data,
and the saved Gaussian uses -C^(-1)(theta-center) from the actual frozen geometry.
This is a conditional score-accuracy sanity check, not a sampler comparison.

{chr(10).join(heur)}

The LEDH score loses to all three simple scores at the low-Q probes p01, p05,
p25 and the pooled mean. That vetoes a claim of uniformly adequate force
approximation over the visited region. At the other four probes it has smaller
observed score error. Neither result establishes an HMC efficiency ranking.
No controls were tuned to these comparisons.

## Decision and inference status

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| No measured M13 derivative repair from this merge | FP64 changes below FD resolution at all 8 probes | No numerical, scope or source veto | Sparse fixed design; other regions unchecked | Preserve this completed before/after result | Global equality of algorithms |
| Own LEDH derivative agrees with its finite likelihood | Maximum component error below 2.6e-6 and refining | All finite and status checks passed | FP64 reference approximates the real finite algorithm | Treat derivative debugging separately from likelihood approximation | Exact model likelihood or full-domain correctness |
| Low-Q proposal/acceptance mismatch remains | Large resolved score disagreement | Simple-score sanity check fails at 4 probes | Finite acceptance target and only 8 visited states | Within the broader surrogate-HMC campaign, investigate low-Q likelihood approximation before retuning proposals | Sole cause of sticky chains or invalid MH correction |
| FP32 merge benefit unsupported | Small, mixed primary-run changes | Cross-process repeatability limits attribution | Only one additional process per version | Keep differences descriptive | Default change, performance ranking or posterior certification |

| Inference item | Status |
|---|---|
| Hard veto screen | All numerical/reference/source checks pass; adequate-force claim fails conditional simple-score comparison |
| Statistically supported ranking | None; deterministic fixed-probe comparisons do not rank sampler performance |
| Descriptive-only differences | FP32 pre/post changes, repeat variability, runtime and conditional force errors |
| Default-readiness | Not evaluated; no runtime/default changed |
| Next evidence needed | A target-specific likelihood repair and downstream surrogate-HMC evaluation; a reproducible FP32 benefit requires process/compilation replication |

The next investigation named above belongs to the existing surrogate-HMC goal.
It was not launched here: this authorized task was a bounded score measurement,
and the previous sampling allocation had expired. A new long sampling run is
neither needed to answer this question nor implied by this result.

## Execution, review and limits

Numerical work used {total/60:.2f} aggregate minutes including CPU preparation,
all primary workers, the failed packaging attempt and both repeat workers,
within the 60-minute cap. The actual GPU was NVIDIA GeForce RTX 5080
(CUDA_VISIBLE_DEVICES=1), with memory growth configured and verified before
initialization. TensorFlow 2.20.0-dev0+selfbuilt / TFP 0.25.0, XLA batch 4,
FP32 with TF32 enabled, and FP64 diagnostic references were recorded.
Peak main-worker TensorFlow allocator usage was 274369024 bytes in FP32 and
544613376 bytes in FP64. These timings are not a performance comparison.

One post-sync snapshot import failed because the Python-only source copy lacked
an existing compiled Sylvester library. Copying that unchanged library repaired
packaging; the original failure and library SHA are retained. A postprocessing
parity assertion then exposed one FP32 ULP from reconstructing constant division
as scalar division rather than the observed XLA rounded-reciprocal operation.
The analysis was corrected and all eight actual runtime forces matched exactly
in both versions. Experimental settings and numerical source were unchanged.

Review checked the saved consumer scope, source/import closure, once-rounded
common inputs, derivative step ordering/refinement, finite/status flags, force
sign and prior, L^T coordinate transformation, absolute-coefficient propagation
of derivative-resolution bounds, and all-eight runtime force parity. The
machine-readable [final audit](final-audit.json) records the checks.

The strongest alternative explanation is that these probes miss a region where
a shared fix matters, or that the finite importance target itself differs from
the integrated model. Both remain possible. A resolved derivative failure at
additional fixed probes, or a better independent likelihood reference showing
the importance approximation is inaccurate here, would alter the interpretation.
The weakest evidence is broad coverage: only eight visited states and two FP32
processes per version were checked. No new chain, acceptance, R-hat, MCMC ESS,
or posterior comparison was measured.

Artifacts: [structured result](report.json), [run manifest](run-manifest.json),
[scope parity](saved-chain-scope-check.json), [source call-chain impact](source-impact-check.json),
[analysis](analyze.py), [FP32 repeat diagnostics](observability-01/manifest.json).
"""
(R/"result.md").write_text(note)
audit=dict(status="PASS",scope=scope,source_import_closure_checked=True,
    imported_module_counts={d["arm"]+"-"+d["dtype"]:len(d["imported_source_hashes"]) for _,d in workers},
    all_points_valid=s["all_points_valid"],all_fd_checks_stable=s["all_fd_checks_stable"],
    endpoint_values_and_derivatives_unchanged=True,all_eight_runtime_force_parity=s["fp32_repeatability"],
    gpu_memory_growth_verified=True,aggregate_seconds_including_cpu_preparation=total,
    budget_seconds=3600,failed_packaging_attempts=1,statistical_ranking_supported=False,
    within_process_repeatability_checked=False,true_integrated_likelihood_score_checked=False,
    interpretation="No measured M13 derivative repair; large conditional proposal-versus-acceptance score mismatch remains.")
write(R/"final-audit.json",audit)
if not (R/"run-manifest-primary-complete.json").exists():
    (R/"run-manifest-primary-complete.json").write_bytes((R/"run-manifest.json").read_bytes())
manifest["workers"]=base_workers
for arm,row in zip(("before","after"),obmanifest["workers"]):
    manifest["workers"].append(row|dict(role="fp32_repeatability_observability",log=str(R/"observability-01"/(arm+".log")),
        output=str(R/"observability-01"/(arm+".json")),timeout_seconds=300))
manifest.update(status="COMPLETED",git_branch=sources["branch"],
    provenance_note="Git HEAD alone is insufficient: frozen dirty source inventories and verified imports record the executed code.",
    cpu_preparation=dict(command=["/home/chakwong/anaconda3/envs/tftwogpu/bin/python",str(R/"measure.py"),"--prepare"],
        wall_seconds=common["wall_seconds"],environment=common["environment"],gpu_intentionally_hidden=True),
    aggregate_numerical_worker_seconds=total,wall_seconds=total,
    wall_seconds_definition="Sum of numerical worker subprocess wall times plus CPU preparation; not elapsed calendar duration",
    remaining_worker_budget_seconds=3600-total,result_written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    script_sha256=sha(R/"measure.py"),
    original_script_sha256=sha(R/"measure-attempt1.py"),
    numerical_script_versions={k:sha(R/k) for k in ("measure.py","measure-attempt1.py","resume.py","observe.py","analyze.py","finalize.py")},
    gpu_environment=workers[0][1]["environment"],data_version=common["data"]["observation_checksum"],
    hardware="NVIDIA GeForce RTX 5080, CUDA_VISIBLE_DEVICES=1, memory growth verified, XLA; CPU-only preparation",
    result_artifacts={str(p.relative_to(R)):sha(p) for p in (
        R/"result.md",R/"report.json",R/"final-audit.json",R/"inputs.json",R/"source-snapshots.json",
        R/"source-impact-check.json",R/"saved-chain-scope-check.json",R/"runtime-library-snapshot.json",
        *[p for p,d in workers],R/"observability-01"/"before.json",R/"observability-01"/"after.json")},
    scope_controls=common["controls"],new_hmc_transitions=0,
    analysis_repairs=["FP32 constant-division reconstruction changed to observed XLA rounded reciprocal; exact runtime-force parity at all eight probes",
        "Whitened derivative-resolution bound propagates absolute L coefficients to avoid cancellation"],
    conclusions=dict(main_fix_improves_m13_score_at_tested_probes=False,own_finite_likelihood_derivative_agreement=True,
        finite_acceptance_likelihood_disagreement_remains=True,statistical_ranking_supported=False))
write(R/"run-manifest.json",manifest)
checkpoint=ROOT/"docs/plans/artifacts/surrogate-hmc-all-models-20260919-review-tests/checkpoint.md"
checkpoint.write_text(f"""# Active M13 score measurement — completed 2026-10-04

Question: did main synchronization improve M13 score accuracy?
Stage: requested measurement COMPLETE; all workers exited. No HMC transitions.
Result: {str((R/"result.md").relative_to(ROOT))}
Plan: docs/plans/surrogate-hmc-m13-score-before-after-2026-10-04.md.

Checked eight frozen visited-state probes, exact saved force scope and source
hashes, FP32 actual runtime wrapper plus FP64 three-step finite differences.
Pre/post FP64 score component change <=2.94e-12. Own-likelihood derivative
errors <=2.57e-6 with quadratic refinement in both versions. Main has not
materially repaired the M13 score at these points. M13 does not call KSC's
observation callback. All validity/reference/source checks passed.

Remaining issue: large LEDH proposal versus frozen importance-acceptance score
mismatch in low Q. At p01: LEDH64 (127.35414,63.65716) versus acceptance
(5.65158,2.23317); updated FP32 (127.76327,64.13024). Simple score baselines
are closer at p01,p05,p25,mean. No sampler-performance or convergence claim.
FP32 cross-process changes <1% in relative norm; no reproducible merge
benefit established. Existing FP32 default unchanged.

Numerical budget: {total:.2f}/3600 seconds including CPU preparation and failed
packaging attempt. One library-copy repair; all six numerical jobs completed.
Results and source snapshots immutable by convention; final manifest/audit saved.
Original stash 2ca7b4f06b623cdfe694ed67c2fde89449cecf37 and unrelated dirty work
preserved. HEAD a55c6be7a16e9401242ee3449248fa347eb58d90 on
ledh-graceful-failure-execution-20260917 plus local modifications.
Sampling allocation remains expired; no sampling job active.

Next action: present this completed measurement. Broader campaign remains
unfinished; a later authorized repair should investigate low-Q likelihood
approximation, not assume the main score fix repaired M13. Do not restart
completed workers or launch a new long HMC run from the old deadline.
""")
print(json.dumps(dict(result=str(R/"result.md"),audit="PASS",aggregate_minutes=total/60,
    worker_count=len(manifest["workers"]),new_hmc_transitions=0)))
