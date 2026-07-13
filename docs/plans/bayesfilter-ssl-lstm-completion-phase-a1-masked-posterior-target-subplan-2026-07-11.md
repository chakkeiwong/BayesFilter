# Phase A1 Subplan: Reusable Masked SSL-LSTM Posterior Target

Date: 2026-07-11

Status: `OWNER_AUTHORIZED_LIFECYCLE_REPAIR_REVIEW_CANDIDATE`

## Phase Objective

Extract the A0-locked four-coordinate scalar SSL-LSTM SVD-UKF posterior into a
production-owned TensorFlow module with a typed parameter mask, stable target
and adapter signatures, graph-native analytic value/score surfaces, and XLA JIT
enabled by default, without changing the historical estimand.

A1 is an engineering-preservation phase. It does not run HMC, fit geometry,
train NeuTra, forecast, calibrate predictive equivalence, or establish posterior
correctness or scientific validity.

## Entry Conditions Inherited From A0

All are conjunctive:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md`
  has final status `PASSED_FOR_A1_IMPLEMENTATION_ONLY`.
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json`
  has exact file SHA-256
  `1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383`.
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json`
  has exact file SHA-256
  `2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517`.
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`
  has exact file SHA-256
  `e8bb6e8dbc861f9c63982e8ea4f67d2cfa4c6cf413ab9e5d5ec5763858af6954`.
- The exact scoped A0 integrity preflight below exits `0`, preserves the exact
  accepted target-lock bytes, and verifies immutable aggregate
  `6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f`
  and signature aggregate
  `af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc`,
  and rehashes every A1 target-critical dependency against the exact accepted
  A0 dependency manifest.
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-result-final-codex-substitute-review-2026-07-11.md`
  ends `VERDICT: AGREE` and binds the exact current SHA-256 of the final A0
  result.
- This subplan has an independent bounded material read-only review with no
  unresolved finding. Exact record
  `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md`
  ends `VERDICT: AGREE` and binds the exact current SHA-256 of this subplan.
- The literal golden payload has its own independent semantic review against A0
  and the A1 contract. Exact record
  `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md`
  ends `GOLDEN VERDICT: AGREE` and binds exact golden file SHA-256
  `04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34`,
  the current A1 subplan SHA-256, A0 target-lock SHA-256
  `1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383`,
  and A0 dependency-manifest SHA-256
  `2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517`.
  Digest self-consistency alone is not semantic acceptance.
- Any byte change to a reviewed result, subplan, or golden file invalidates its
  review and returns A1 to this entry gate.
- The immutable A0 anchor commit is
  `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`. The actual A1 evidence-run
  commit may differ only when the anchor is its Git ancestor and every committed
  path in `anchor..evidence_run_commit` is disjoint from the exact protected and
  A1-owned sets below. The preflight records both commits. Any other `HEAD`
  movement is a continuation veto requiring reviewed reconciliation.
- Unrelated dirty work remains user-owned and must not be reset, reformatted,
  staged, committed, or overwritten. The user's 2026-07-12 instruction that
  another agent is working in a different lane authorizes unrelated concurrent
  changes to remain observable but non-vetoing for A1. It does not authorize
  drift in the protected target-critical rows or writes into the A1-owned set.
- No historical benchmark script is imported by production code.

If an A0 immutable member changes, A1 does not patch around it; stop and reopen
the A0 checkpoint.

First create `/tmp/bayesfilter-ssl-lstm-a1-runtime/`, then run this exact
non-TensorFlow entry preflight. It checks commit ancestry/scope, fixed artifacts, hash-bound
review records, the accepted A0 lock identities, and the current bytes of every
target-critical dependency. It partitions all 51 unique A0 manifest paths into
23 protected rows and 28 exact exclusions. Every excluded row retains its A0
hash and roles plus a reviewed reason; the partition must be exhaustive and
disjoint. Excluded package-initialization-only HMC/NeuTra/runtime modules are
exercised by imports and parity tests, while historical context rows remain
non-promoting and unused. The preflight is a one-time writer. It must exit `0`,
emit `status=a1_entry_scoped_integrity_verified`, and write deterministic
`a0-entry-verification.json` directly before the scoped boundary is created.
After the boundary binds its exact SHA-256, this writer must never be rerun;
all later entry checks are strict read-only verification of the frozen bytes:

```bash
mkdir -p /tmp/bayesfilter-ssl-lstm-a1-runtime

/home/ubuntu/anaconda3/envs/tfgpu/bin/python -c '
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd().resolve()
def require(condition,message):
    if not condition: raise RuntimeError(message)
def digest(relative): return hashlib.sha256((root/relative).read_bytes()).hexdigest()
def reject(value): raise ValueError("nonfinite JSON constant")
def pairs(rows):
    out={}
    for key,value in rows:
        if key in out: raise ValueError("duplicate JSON key")
        out[key]=value
    return out
def strict(relative):
    return json.loads((root/relative).read_text(),parse_constant=reject,object_pairs_hook=pairs)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")
def committed_history_paths(anchor,head):
    if anchor==head: return []
    commits=subprocess.check_output(["git","rev-list","--reverse",anchor+".."+head],cwd=root,text=True).splitlines()
    paths=set()
    for commit in commits:
        raw=subprocess.check_output(["git","diff-tree","--root","-m","--no-commit-id","--name-only","-r","-z",commit],cwd=root)
        paths.update(os.fsdecode(row) for row in raw.split(b"\0") if row)
    return sorted(paths)
def agreed(record,target,final_line):
    text=(root/record).read_text()
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    require(f"Reviewed path: `{target}`" in text,f"review path binding missing: {record}")
    require(f"Reviewed SHA-256: `{digest(target)}`" in text,f"review hash binding missing: {record}")
    require(lines and lines[-1]==final_line,f"review verdict missing: {record}")
fixed={
 "docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md":"de891cf98e600acf302ee483ad9357c4e1dee71079a3a28cca3354ffabe11193",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-result-final-codex-substitute-review-2026-07-11.md":"17a35da0b55cf9fc78e6b26eddda6d47d606d6797bc8724a3277df8383364e15",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json":"1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json":"2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517",
 "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py":"e8bb6e8dbc861f9c63982e8ea4f67d2cfa4c6cf413ab9e5d5ec5763858af6954",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json":"04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json":"2ec3b605266fe652fc452d58d483914cea71f9ce845345c685d4669cfcf848be",
 "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py":"fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28",
}
for path,expected in fixed.items(): require(digest(path)==expected,f"fixed hash drift: {path}")
a0="docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md"
a1="docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md"
gold="docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json"
require("Status: `PASSED_FOR_A1_IMPLEMENTATION_ONLY`" in (root/a0).read_text(),"A0 final status missing")
agreed("docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-result-final-codex-substitute-review-2026-07-11.md",a0,"VERDICT: AGREE")
agreed("docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md",a1,"VERDICT: AGREE")
gold_review="docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md"
agreed(gold_review,gold,"GOLDEN VERDICT: AGREE")
review_text=(root/gold_review).read_text()
for label,path in (("Contract",a1),("A0 target lock","docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"),("A0 dependency manifest","docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json")):
    require(f"{label} path: `{path}`" in review_text,f"golden semantic-review path binding missing: {label}")
    require(f"{label} SHA-256: `{digest(path)}`" in review_text,f"golden semantic-review hash binding missing: {label}")
approval_path="docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md"
approval_text=(root/approval_path).read_text()
approval_start="<!-- BEGIN A1 SCOPED CONCURRENT-LANE AUTHORIZATION -->"
approval_end="<!-- END A1 SCOPED CONCURRENT-LANE AUTHORIZATION -->"
require(approval_text.count(approval_start)==1 and approval_text.count(approval_end)==1,"scoped authorization markers missing or duplicated")
approval_section=approval_text[approval_text.index(approval_start):approval_text.index(approval_end)+len(approval_end)]
approval_section_sha256=hashlib.sha256(approval_section.encode("utf-8")).hexdigest()
require(approval_section_sha256=="93d6be42737a9c5d34b3cee638081b83d7d4733a7ad31927ab0f355702834b4d","scoped concurrent-lane authorization drift")
lock=strict("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json")
manifest=strict("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json")
require(lock["schema_version"]=="bayesfilter.ssl_lstm_completion.phase_a0_target_lock.v1","A0 lock schema drift")
require(lock["immutable_attempt_fingerprint"]["aggregate_sha256"]=="6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f","A0 immutable aggregate drift")
require(lock["signatures"]["aggregate_sha256"]=="af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc","A0 signature aggregate drift")
require(lock["signatures"]["target_semantic_sha256"]=="549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e","A0 target signature drift")
require(lock["target_semantics"]["observations"]["raw_sha256"]=="aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98","A0 observation identity drift")
require(lock["target_semantics"]["full_fixture"]["raw_sha256"]=="33b0814b86c5875e6746150762b8ae3b655e5bbcaa0bfd8df51488783bcb601f","A0 fixture identity drift")
protected_paths=(
 "bayesfilter/__init__.py","bayesfilter/diagnostics.py",
 "bayesfilter/inference/__init__.py",
 "bayesfilter/inference/batched_value_score.py","bayesfilter/inference/posterior_adapter.py",
 "bayesfilter/linear/__init__.py",
 "bayesfilter/linear/dtypes_tf.py","bayesfilter/linear/qr_factor_tf.py",
 "bayesfilter/linear/svd_factor_tf.py","bayesfilter/linear/types_tf.py",
 "bayesfilter/nonlinear/cut_tf.py","bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py",
 "bayesfilter/nonlinear/fixed_sgqf_tf.py","bayesfilter/nonlinear/sigma_points_tf.py",
 "bayesfilter/nonlinear/ssl_lstm_protocol.py","bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py",
 "bayesfilter/nonlinear/ssl_lstm_zhaocui_fixed_adapter.py","bayesfilter/nonlinear/ssl_lstm_zhaocui_hmc_minimal.py",
 "bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py","bayesfilter/results_tf.py",
 "bayesfilter/structural.py","bayesfilter/structural_tf.py",
 "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py",
)
excluded_reasons={
 "bayesfilter/inference/backend_parity.py":"package_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/inference/fixed_transport_hmc.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/fixed_transport_hmc_grid_policy.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/fixed_transport_hmc_tuning.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/generic_hmc_tuning.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/hmc.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/hmc_budget_ladder.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/hmc_diagnostics.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/hmc_kernel_tuning.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/hmc_tuning.py":"hmc_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/mass_matrix.py":"geometry_or_hmc_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/inference/neutra_artifacts.py":"neutra_namespace_only_forbidden_in_a1",
 "bayesfilter/inference/quadratic_geometry.py":"historical_geometry_diagnostic_symbol_imported_but_not_called_by_target_constructor_or_value_score",
 "bayesfilter/inference/quadratic_map_covariance.py":"geometry_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/inference/target_failure_policy.py":"package_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/nonlinear/__init__.py":"a1_owned_lazy_export_file_intentionally_supersedes_a0_bytes_and_is_reviewed_as_a1_source",
 "bayesfilter/runtime/__init__.py":"runtime_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/runtime/device_policy.py":"runtime_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/runtime/runner.py":"runtime_namespace_only_not_called_by_a1_target_value_score",
 "bayesfilter/runtime/selection.py":"runtime_namespace_only_not_called_by_a1_target_value_score",
 "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py":"historical_sampler_geometry_context_only_not_an_a1_target_input",
 "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json":"historical_result_context_only_replaced_by_live_all_ten_point_replay",
 "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json":"historical_sampler_geometry_context_only_not_an_a1_target_input",
 "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json":"historical_hmc_diagnostic_context_only_forbidden_as_a1_evidence",
 "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json":"historical_reference_context_only_forbidden_as_a1_evidence",
 "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json":"historical_reference_context_only_forbidden_as_a1_evidence",
 "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json":"historical_reference_context_only_forbidden_as_a1_evidence",
 "docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json":"historical_reference_context_only_forbidden_as_a1_evidence",
}
owned_paths=(
 "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py","bayesfilter/nonlinear/__init__.py",
 "tests/test_ssl_lstm_posterior_tf.py",
 "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py",
 "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md",
 "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md",
 "docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md",
 "docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md",
 "docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md",
 "docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md",
 "docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-scoped-boundary.json",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.log",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json",
 "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.log",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-codex-substitute-review-2026-07-11.md",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-implementation-codex-substitute-review-2026-07-11.md",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md",
 "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-2026-07-11.md",
)
manifest_rows={}
for collection in ("critical_roots","runtime_loaded_local_dependencies","historical_inputs"):
    for row in manifest[collection]:
        prior=manifest_rows.setdefault(row["path"],{"sha256":row["sha256"],"roles":set()})
        require(prior["sha256"]==row["sha256"],"manifest hash conflict: "+row["path"])
        prior["roles"].add(row["role"])
require(len(manifest_rows)==51,"unexpected unique A0 manifest path count")
require(not (set(protected_paths)&set(excluded_reasons)),"protected/excluded manifest partition overlaps")
require(set(protected_paths)|set(excluded_reasons)==set(manifest_rows),"protected/excluded manifest partition is not exhaustive")
protected=[]
for path in protected_paths:
    require(path in manifest_rows,f"protected path absent from A0 manifest: {path}")
    expected=manifest_rows[path]["sha256"]
    require(digest(path)==expected,f"protected dependency drift: {path}")
    protected.append({"path":path,"sha256":expected,"roles":sorted(manifest_rows[path]["roles"])})
protected.sort(key=lambda row:row["path"])
excluded=[{"path":path,"accepted_a0_sha256":manifest_rows[path]["sha256"],"roles":sorted(manifest_rows[path]["roles"]),"exclusion_reason":excluded_reasons[path]} for path in sorted(excluded_reasons)]
anchor_commit="a644d29c5c2fd09a0deb3a7b5212799ff1fcb163"
evidence_run_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
require(subprocess.run(["git","merge-base","--is-ancestor",anchor_commit,evidence_run_commit],cwd=root).returncode==0,"A0 anchor is not an ancestor of evidence-run commit")
committed_paths=committed_history_paths(anchor_commit,evidence_run_commit)
forbidden_committed=sorted(set(committed_paths)&(set(protected_paths)|set(owned_paths)))
require(not forbidden_committed,"committed protected/A1-owned path drift: "+repr(forbidden_committed))
documents={"schema_version":"bayesfilter.ssl_lstm_completion.phase_a1_entry_documents.v2","a0_anchor_commit":anchor_commit,"fixed_sha256":dict(sorted(fixed.items())),"reviewed_target_sha256":{path:digest(path) for path in sorted((a0,a1,gold))},"review_sha256":{path:digest(path) for path in sorted(("docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-result-final-codex-substitute-review-2026-07-11.md","docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md",gold_review))},"approval_boundary":{"path":approval_path,"section_sha256":approval_section_sha256}}
integrity={"status":"target_lock_integrity_verified_under_scoped_concurrency","artifact":str((root/"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json").resolve()),"target_lock_file_sha256":digest("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"),"dependency_manifest_file_sha256":digest("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json"),"immutable_aggregate":"6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f","signature_aggregate":"af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc","target_semantic_sha256":"549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"}
payload={"schema_version":"bayesfilter.ssl_lstm_completion.phase_a1_entry_verification.v2","entry_documents":documents,"target_lock_integrity":integrity,"protected_dependency_rows":protected,"protected_dependency_aggregate_sha256":hashlib.sha256(canonical(protected)).hexdigest(),"excluded_dependency_rows":excluded,"excluded_dependency_aggregate_sha256":hashlib.sha256(canonical(excluded)).hexdigest(),"manifest_partition":{"unique_manifest_path_count":51,"protected_path_count":len(protected),"excluded_path_count":len(excluded),"exhaustive":True,"disjoint":True},"commit_policy":{"a0_anchor_commit":anchor_commit,"entry_checked_commit":evidence_run_commit,"committed_paths_since_anchor":sorted(committed_paths),"forbidden_committed_paths":[]},"concurrent_lane_policy":"unrelated_changes_observed_not_vetoing_protected_or_a1_owned_changes_veto"}
output=Path("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json")
output.write_text(json.dumps(payload,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)+"\n")
print("status=a1_entry_scoped_integrity_verified")
'
```

The accepted A0 verifier remains the authority that created and closed the
immutable A0 lock. It is not rerun live in A1 because its discovery-only package
closure includes HMC modules that the user-authorized concurrent lane is
changing and that A1 does not call numerically. A1 does not reinterpret that
historical verifier failure as target drift. Current target identity is instead
tested by the exact lock/file checks above plus all-ten-point historical
value/score replay below. A protected-row mismatch is a continuation veto; an
unrelated namespace-row mismatch is not.

## Locked Estimand

| Field | Binding A0 value |
| --- | --- |
| Observations | `float64[30,1]`, raw SHA-256 `aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98` |
| Full chart | Exact ordered 24-name `SSLLSTMStaticConfig` chart |
| Full fixture | `float64[24]`, raw SHA-256 `33b0814b86c5875e6746150762b8ae3b655e5bbcaa0bfd8df51488783bcb601f` |
| Free mask | names `(latent_mean_weight.0.0, latent_mean_bias.0, observation_weight.0.0, observation_bias.0)`, indices `(12,13,14,15)` |
| Prior | Unnormalized Gaussian log kernel centered at `(0.35,-0.08,0.65,0.05)` with standard deviation `4.0`; do not add a normalizing constant |
| Filter | Historical eigenderivative `tf_svd_ukf_score` route, not the principal-square-root route |
| Dtype | `float64` only in A1 |
| Target signature | `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |

The Phase 2S center/scale/factor remain sampler context and test-point
provenance only. They do not enter target semantics.

Historical parity at every finite A1 test point uses the following exact rule.
For historical scalar `h` and A1 scalar `a`, both must be finite and
`abs(a-h) <= 8*(2**-52)*max(1,abs(a),abs(h))`. For historical score `h[4]` and
A1 score `a[4]`, every coordinate must be finite and
`max_i(abs(a_i-h_i)) <= 8*(2**-52)*max(1,max_i(abs(a_i)),max_i(abs(h_i)))`.
The literal scale floor `1` controls zero and near-zero values. The historical
operand comes only from the hash-pinned eager historical constructor; the A1
operand comes only from the eager production finite branch.

## Exact Write Set

A1 may create or edit only:

- `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py`;
- `bayesfilter/nonlinear/__init__.py`, only for lazy exports of the reviewed A1
  public/internal symbols;
- `tests/test_ssl_lstm_posterior_tf.py`;
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-scoped-boundary.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.log`;
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json`,
  which is a pre-implementation reviewed planning contract and becomes
  immutable once this subplan review is accepted;
- this subplan and
  `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-codex-substitute-review-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-implementation-codex-substitute-review-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md`;
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-2026-07-11.md`;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md`
  after final A1 CPU/GPU evidence passes.

The only permitted `bayesfilter/nonlinear/__init__.py` additions are lazy
exports for `SSLLSTMParameterMask`, `SSLLSTMPosteriorConfig`,
`SSLLSTMPosteriorTarget`, and `locked_ssl_lstm_posterior_target`.
Temporary bytecode/test/compiler state is redirected to
`/tmp/bayesfilter-ssl-lstm-a1-pycache/` and
`/tmp/bayesfilter-ssl-lstm-a1-runtime/`; it is non-evidence scratch, not a
repository write. Pytest caching is disabled.

The pre-existing
`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json`
is A1-owned read-only legacy context, not a mutable write-set member. Its exact
SHA-256 remains
`2ec3b605266fe652fc452d58d483914cea71f9ce845345c685d4669cfcf848be`;
A1 must not edit or replace it. The preflight and v2 boundary enforce only its
file hash and regular-file identity; its historical whole-worktree rows and
aggregate are never compared with the current concurrent worktree and do not
enter any promotion criterion.

After the amended plan and golden reviews agree, and before any further
production/test/harness edit, create the scoped boundary at the new exact v2
path below. The stale whole-repository inventory remains untouched as
superseded legacy context and is not a v2 verification input.

The artifact binds the literal A1-owned set, immutable A0 anchor, exact
boundary-checked commit, complete protected/excluded partition from
`a0-entry-verification.json`, immutable A1 inputs, and
an explanatory snapshot of the unrelated Git index/worktree. Verification
requires a safe anchor-to-current commit path, protected rows/aggregate,
exhaustive partition, immutable inputs, owned list, schema, and canonical
artifact aggregate. It deliberately does not
require the unrelated Git index or porcelain snapshot to remain equal: those
fields document concurrent activity and are explanatory only. A protected-row
change, an unexpected file under the Phase A1 artifact namespace, or a write to
an A1-owned path not attributable to the current supervised phase step is a
continuation veto. Other changes are observed but non-vetoing.

The concurrency threat model is the user-authorized cooperative other lane:
that lane may change unrelated paths but must not touch an A1-owned path. The
boundary command has sole authorized ownership of its new output path and no
authorized consumer reads it until the command succeeds. It acquires the final
path once with `O_CREAT|O_EXCL|O_NOFOLLOW`, writes and verifies through the same
descriptor, requires regular-file/single-link device-inode identity through a
post-write critical snapshot, rereads bytes and size after that snapshot, and
performs a final `lstat` identity check. It never renames, replaces, follows, or
automatically unlinks a pathname. Creation failure or any identity/snapshot
mismatch leaves a blocker for reviewed repair; it does not clean up by path.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -c '
import hashlib,json,os,stat,subprocess
from pathlib import Path
root=Path.cwd().resolve()
def reject(value): raise ValueError("nonfinite JSON constant")
def pairs(rows):
    out={}
    for key,value in rows:
        if key in out: raise ValueError("duplicate JSON key")
        out[key]=value
    return out
def strict(path): return json.loads(path.read_text(),parse_constant=reject,object_pairs_hook=pairs)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def committed_history_paths(anchor,head):
    if anchor==head: return []
    commits=subprocess.check_output(["git","rev-list","--reverse",anchor+".."+head],cwd=root,text=True).splitlines()
    paths=set()
    for commit in commits:
        raw=subprocess.check_output(["git","diff-tree","--root","-m","--no-commit-id","--name-only","-r","-z",commit],cwd=root)
        paths.update(os.fsdecode(row) for row in raw.split(b"\0") if row)
    return sorted(paths)
def path_state(relative):
    path=root/relative
    if not path.exists() and not path.is_symlink(): return {"path":relative,"kind":"missing","mode":None,"sha256":None}
    if path.is_symlink(): kind="symlink"; raw=os.fsencode(os.readlink(path))
    elif path.is_file(): kind="regular"; raw=path.read_bytes()
    elif path.is_dir(): kind="directory"; raw=b""
    else: kind="other"; raw=b""
    return {"path":relative,"kind":kind,"mode":format(stat.S_IMODE(path.lstat().st_mode),"04o"),"sha256":hashlib.sha256(raw).hexdigest()}
owned_exact=sorted({
"bayesfilter/nonlinear/ssl_lstm_posterior_tf.py",
"bayesfilter/nonlinear/__init__.py",
"tests/test_ssl_lstm_posterior_tf.py",
"docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py",
"docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md",
"docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md",
"docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md",
"docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md",
"docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md",
"docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md",
"docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-scoped-boundary.json",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.log",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json",
"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.log",
"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-subplan-codex-substitute-review-2026-07-11.md",
"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-codex-substitute-review-2026-07-11.md",
"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-golden-signatures-current-contract-codex-substitute-review-2026-07-11.md",
"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-implementation-codex-substitute-review-2026-07-11.md",
"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md",
"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-2026-07-11.md",
})
entry_path=root/"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json"
entry=strict(entry_path)
assert entry["schema_version"]=="bayesfilter.ssl_lstm_completion.phase_a1_entry_verification.v2"
artifact_dir=root/"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1"
allowed_artifact_names={"a0-entry-verification.json","pre-run-outside-write-set-inventory.json","pre-run-scoped-boundary.json","golden-signatures.json","cpu-reference.json","cpu-reference.log","gpu-xla-canary.json","gpu-xla-canary.log"}
output=artifact_dir/"pre-run-scoped-boundary.json"
assert not os.path.lexists(output)
immutable_inputs=[
 {"path":"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/a0-entry-verification.json","sha256":digest(entry_path)},
 {"path":"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json","sha256":"04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34"},
 {"path":"docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/pre-run-outside-write-set-inventory.json","sha256":"2ec3b605266fe652fc452d58d483914cea71f9ce845345c685d4669cfcf848be"},
]
for row in immutable_inputs:
    state=path_state(row["path"])
    assert state["kind"]=="regular" and state["sha256"]==row["sha256"],state
approval_binding=entry["entry_documents"]["approval_boundary"]
approval_text=(root/approval_binding["path"]).read_text()
approval_start="<!-- BEGIN A1 SCOPED CONCURRENT-LANE AUTHORIZATION -->"
approval_end="<!-- END A1 SCOPED CONCURRENT-LANE AUTHORIZATION -->"
approval_section=approval_text[approval_text.index(approval_start):approval_text.index(approval_end)+len(approval_end)]
assert hashlib.sha256(approval_section.encode("utf-8")).hexdigest()==approval_binding["section_sha256"]
anchor_commit=entry["commit_policy"]["a0_anchor_commit"]
def critical_snapshot():
    checked_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    assert subprocess.run(["git","merge-base","--is-ancestor",anchor_commit,checked_commit],cwd=root).returncode==0
    committed_paths=committed_history_paths(anchor_commit,checked_commit)
    forbidden=sorted(set(committed_paths)&(set(row["path"] for row in entry["protected_dependency_rows"])|set(owned_exact)))
    assert not forbidden,forbidden
    protected_state=[]
    for row in entry["protected_dependency_rows"]:
        state=path_state(row["path"])
        assert state["kind"]=="regular" and state["sha256"]==row["sha256"],state
        protected_state.append(state)
    bound_rows=[]
    documents=entry["entry_documents"]
    for collection in ("fixed_sha256","reviewed_target_sha256","review_sha256"):
        for path,expected in sorted(documents[collection].items()):
            state=path_state(path)
            assert state["kind"]=="regular" and state["sha256"]==expected,state
            bound_rows.append({"binding_collection":collection,**state})
    current_approval_text=(root/approval_binding["path"]).read_text()
    current_approval_section=current_approval_text[current_approval_text.index(approval_start):current_approval_text.index(approval_end)+len(approval_end)]
    current_approval_sha256=hashlib.sha256(current_approval_section.encode("utf-8")).hexdigest()
    assert current_approval_sha256==approval_binding["section_sha256"]
    namespace=sorted(path.name for path in artifact_dir.iterdir() if path.name!=output.name)
    assert not (set(namespace)-allowed_artifact_names),namespace
    owned_state=[path_state(path) for path in owned_exact if path!=output.relative_to(root).as_posix()]
    return {"boundary_checked_commit":checked_commit,"committed_paths_since_anchor":committed_paths,"protected_state":protected_state,"bound_document_state":bound_rows,"approval_section_sha256":current_approval_sha256,"artifact_namespace_excluding_boundary":namespace,"owned_state_excluding_boundary":owned_state}
pre_snapshot=critical_snapshot()
porcelain=subprocess.check_output(["git","status","--porcelain=v1","-z"],cwd=root)
unrelated_snapshot={"role":"explanatory_only_not_a_verification_equality","git_index_sha256":hashlib.sha256(subprocess.check_output(["git","ls-files","--stage","-z"],cwd=root)).hexdigest(),"git_porcelain_sha256":hashlib.sha256(porcelain).hexdigest(),"git_porcelain_entry_count":len([row for row in porcelain.split(b"\0") if row])}
projection={"a0_anchor_commit":anchor_commit,"boundary_creation_commit":pre_snapshot["boundary_checked_commit"],"committed_paths_through_boundary_creation":pre_snapshot["committed_paths_since_anchor"],"owned_exact":owned_exact,"initial_owned_state":pre_snapshot["owned_state_excluding_boundary"],"protected_dependency_rows":entry["protected_dependency_rows"],"protected_dependency_aggregate_sha256":entry["protected_dependency_aggregate_sha256"],"excluded_dependency_rows":entry["excluded_dependency_rows"],"excluded_dependency_aggregate_sha256":entry["excluded_dependency_aggregate_sha256"],"manifest_partition":entry["manifest_partition"],"immutable_inputs":immutable_inputs,"approval_section_sha256":pre_snapshot["approval_section_sha256"],"concurrent_lane_policy":entry["concurrent_lane_policy"],"unrelated_snapshot":unrelated_snapshot}
canonical=json.dumps(projection,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")
artifact={"schema_version":"bayesfilter.ssl_lstm_completion.phase_a1_scoped_boundary.v2",**projection,"aggregate_sha256":hashlib.sha256(canonical).hexdigest()}
artifact_bytes=(json.dumps(artifact,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)+"\n").encode("utf-8")
assert hasattr(os,"O_NOFOLLOW")
fd=os.open(output,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
try:
    opened=os.fstat(fd)
    assert stat.S_ISREG(opened.st_mode) and opened.st_nlink==1
    created_identity=(opened.st_dev,opened.st_ino)
    remaining=memoryview(artifact_bytes)
    while remaining:
        written=os.write(fd,remaining)
        assert written>0
        remaining=remaining[written:]
    os.fsync(fd)
    written_state=os.fstat(fd)
    assert stat.S_ISREG(written_state.st_mode) and written_state.st_nlink==1
    assert (written_state.st_dev,written_state.st_ino)==created_identity
    assert written_state.st_size==len(artifact_bytes)
    linked=os.lstat(output)
    assert stat.S_ISREG(linked.st_mode) and linked.st_nlink==1
    assert (linked.st_dev,linked.st_ino)==created_identity
    assert critical_snapshot()==pre_snapshot
    final_opened=os.fstat(fd)
    assert stat.S_ISREG(final_opened.st_mode) and final_opened.st_nlink==1
    assert (final_opened.st_dev,final_opened.st_ino)==created_identity
    assert final_opened.st_size==len(artifact_bytes)
    os.lseek(fd,0,os.SEEK_SET)
    assert os.read(fd,len(artifact_bytes)+1)==artifact_bytes
    final_link=os.lstat(output)
    assert stat.S_ISREG(final_link.st_mode) and final_link.st_nlink==1
    assert (final_link.st_dev,final_link.st_ino)==created_identity
finally:
    os.close(fd)
print(json.dumps({"status":"phase_a1_scoped_boundary_written","protected_dependency_row_count":len(entry["protected_dependency_rows"]),"aggregate_sha256":artifact["aggregate_sha256"]},sort_keys=True,separators=(",",":")))
'
```

### Frozen Entry And Live History Lifecycle

The entry artifact and live Git history have separate roles:

- `a0-entry-verification.json` is written exactly once by the reviewed entry
  preflight. Boundary creation records its exact file SHA-256 in
  `immutable_inputs`. CPU/GPU generation, CPU/GPU verification, and the final
  checkpoint strict-load and reconstruct its v2 schema and projections without
  opening it for write, replacing it, or rerunning the entry writer.
- The entry artifact's `entry_checked_commit` and
  `committed_paths_since_anchor` attest only its creation checkpoint. They are
  not required to equal a later process commit or later live history list.
- Boundary creation and every CPU/GPU generation or verification process
  independently capture current `HEAD` before other work, require the immutable
  A0 anchor to be its ancestor, enumerate every reachable intervening commit
  with the exact `git rev-list` plus per-commit `git diff-tree --root -m`
  algorithm above, and require the accumulated path set to be disjoint from the
  protected and A1-owned sets. Each process recaptures `HEAD` before publication
  or success and requires equality with its opening capture. A change during a
  process fails that process; a safe unrelated change between processes is
  allowed and checked afresh.
- CPU/GPU artifacts bind their live attestation through
  `run_manifest.git_commit` plus
  `contract_checks.scoped_boundary_verified=true`; strict verification
  recomputes the full history rather than trusting that boolean. The A1 result
  records the final-checkpoint commit and accumulated paths. No live-history
  check mutates the frozen entry or boundary artifacts.

This separation preserves deterministic entry bytes while still detecting a
touch-then-restore commit anywhere from the A0 anchor through each evidence or
verification process.

No other `bayesfilter/`, `tests/`, historical benchmark/result, LaTeX, target
lock, prior, model equation, filter implementation, HMC, or NeuTra file is in
the A1 write set. A required A1 edit outside this set is a stop-and-review
event. An edit made independently by the authorized concurrent lane is not an
A1 edit and is non-vetoing unless it changes a protected dependency, an
A1-owned path, or the safe anchor/current commit relation.

## Required Production Contract

### Independent Golden Signatures

The plan, not implementation output, freezes three distinct signature roles.
The binding literal payloads are in
`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json`,
whose exact file SHA-256 is
`04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34`:

1. **A0 semantic target:** `target_signature()` returns exactly
   `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`.
   It is not recomputed from a new wrapper schema.
2. **A1 parameter mask:** schema
   `bayesfilter.ssl_lstm_completion.parameter_mask.v1`, independently derived
   from the A0 lock, has expected SHA-256
   `9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043`.
3. **A1 wrapper/adapter contract:** schema
   `bayesfilter.ssl_lstm_completion.masked_posterior_contract.v1` has expected
   SHA-256
   `004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556`.

Mask and wrapper SHA-256 use the canonicalization object in that file: UTF-8
JSON with sorted keys, separators `,` and `:`, `ensure_ascii=True`,
`allow_nan=False`, and no extra keys. The literal payloads, including field
types, float hex strings, row order, capability, callable contract, and
testing-only restrictions, are binding; prose in this subplan cannot add or
remove a signature member. The golden file itself is strict JSON: duplicate
keys and nonfinite JSON constants are rejected. Its two stored payload digests
must be independently recomputed before implementation and during artifact
verification.

Tests must compare implementation payloads and digests with these independent
goldens and must mutate every semantic component to prove the digest changes.
No test may derive its expected digest by calling the implementation's hash
function on the implementation's own payload.

Exact pre-implementation golden verifier, requiring exit `0` and final status
`golden_signatures_verified`:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -c '
import hashlib,json
from pathlib import Path
p=Path("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json")
def reject(value): raise ValueError("nonfinite JSON constant")
def pairs(rows):
    out={}
    for key,value in rows:
        if key in out: raise ValueError("duplicate JSON key")
        out[key]=value
    return out
value=json.loads(p.read_text(),parse_constant=reject,object_pairs_hook=pairs)
canonical=lambda item: json.dumps(item,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")
assert hashlib.sha256(p.read_bytes()).hexdigest()=="04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34"
assert hashlib.sha256(canonical(value["parameter_mask"]["payload"])).hexdigest()==value["parameter_mask"]["sha256"]=="9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043"
assert hashlib.sha256(canonical(value["masked_posterior_contract"]["payload"])).hexdigest()==value["masked_posterior_contract"]["sha256"]=="004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556"
print("status=golden_signatures_verified")
'
```

The exact ordered nonclaims in both A1 artifacts are:

1. `target extraction and frozen-point engineering canary only`;
2. `not posterior correctness evidence`;
3. `not HMC or NeuTra readiness evidence`;
4. `not predictive equivalence or calibration evidence`;
5. `not target-wide GPU/XLA or performance evidence`;
6. `not public API, default, product, or release readiness evidence`;
7. `not a sampler ranking or scientific claim`.

### `SSLLSTMParameterMask`

A frozen dataclass that binds:

- full ordered parameter names;
- unique ordered free names and full-chart indices;
- fixed full-vector values;
- `float64` dtype and dimensions `24` full / `4` free;
- `embed(free)->full` and `extract(full)->free` TensorFlow operations;
- the exact v1 mask payload and golden SHA-256 above.

Construction rejects duplicate/unknown names, duplicate/out-of-range or
name/index-mismatched indices, wrong full-vector size/dtype, nonfinite fixed
values, and a mask not identical to the locked four-coordinate order. Runtime
surfaces require exact trailing dimension and `float64`; they do not silently
reshape a wrong-sized or lower-precision tensor.

### `SSLLSTMPosteriorConfig`

A frozen dataclass that binds:

- `SSLLSTMStaticConfig(horizon=30, latent_dim=1, hidden_dim=1,
  observation_dim=1, covariance_mode="diagonal")`;
- the parameter mask;
- exact `float64[30,1]` observations and raw hash;
- prior center, standard deviation `4.0`, unnormalized convention;
- `svd_ukf` historical score route and every A0 numerical setting;
- default `jit_compile=True`, backend `tensorflow`, dtype `float64`;
- target scope `ssl_lstm_completion:a1:masked_svd_ukf_four_parameter`;
- A0 target, immutable, and signature aggregates;
- the exact v1 wrapper payload and golden SHA-256 above.

The config rejects a hash/value mismatch, nonfinite observation/prior input,
wrong shapes, wrong filter route, normalized prior, non-XLA default, or any
setting that changes the A0 target. `jit_compile=False` may exist only as an
explicit eager/debug reference option and must be labeled non-default.

### `SSLLSTMPosteriorTarget`

The target exposes:

- `parameter_dim == 4` and the exact four parameter names;
- `full_theta(free)` and `free_theta(full)`;
- `log_prob(free)`, `log_prob_and_grad(free)`, `value(free)`, `score(free)`,
  and `value_and_score(free)`;
- scalar and fixed-static-batch value/score surfaces; batching uses a static
  TensorFlow unroll or a reviewed batch-native route, never a Python/NumPy
  algorithmic loop;
- `adapter_signature()`, `target_signature()`, and a four-dimensional
  `ValueScoreCapability`/manifest payload.

Exact callable semantics:

| Surface | Input | Output | Default execution |
| --- | --- | --- | --- |
| `value_and_score`, `log_prob_and_grad` | `float64[4]` | scalar `float64[]`, score `float64[4]` | enter one cached `tf.function(jit_compile=True, input_signature=[TensorSpec([4],tf.float64)])` |
| `value`, `score`, `log_prob` | `float64[4]` | respectively `[]`, `[4]`, `[]` | delegate to the same compiled scalar value/score program; no second target implementation |
| `batch_value_and_score` | `float64[B,4]`, statically known positive `B` | value `[B]`, score `[B,4]` | cached XLA function per static `B`; mandatory `B in {1,4,10}` tests |
| `eager_debug_value_and_score` | `float64[4]` | `[]`, `[4]` | explicit non-default eager reference only |
| `diagnostic_value_and_score` | scalar or supported batch | value, score, `int32` status `[]` or `[B]` | same target branch; status `0=valid_finite`, `1=nonfinite_input_reject` |

`value == log_prob == value_and_score[0] == log_prob_and_grad[0]` and
`score == value_and_score[1] == log_prob_and_grad[1]` under the same execution
mode. `log_prob` is built with the repository
`reviewed_value_score_target_fn` custom-gradient pattern. A `GradientTape`
gradient of `log_prob` must equal the declared analytic score for valid and
reject branches; autodiff through the filtering recursion remains forbidden.
Dynamic/unknown `B`, rank other than `1`/`2`, wrong trailing dimension, and
non-`float64` inputs are loud errors. A config boolean alone is not the XLA
default: default public calls must demonstrably invoke the compiled functions.

The implementation embeds four free values into the locked 24-vector, calls
`tf_ssl_lstm_svd_ukf_score` with the exact A0 settings, gathers the four score
coordinates, and adds the exact unnormalized prior value/score. It must not
import benchmark code, call NumPy in an algorithmic or gradient-bearing path,
use `tf.py_function`, switch to `tf_principal_sqrt_ukf_score`, add a transform
Jacobian, or normalize the prior.

The historical `SSLLSTMAdapterProtocol` describes the full 24-parameter model
and must not be misrepresented as the four-dimensional HMC coordinate contract.
The A1 target must publish a separate mask-specific stable adapter signature and
capability with `parameter_dim=4` while retaining the full chart inside target
provenance.

### Nonfinite-Input Reject Behavior

The A0 target is locked on finite proposals. A1 does not add a plateau for a
finite proposal whose filter fails. Shape, dtype, configuration, programmer
errors, and any finite-proposal filter exception/nonfinite output remain loud
phase blockers.

The wrapper contract includes exact schema
`bayesfilter.ssl_lstm_completion.nonfinite_input_reject.v1`, scope
`nonfinite_input_only`, `fallback_log_prob_hex=-0x1.249ad2594c37dp+332`
(`-1e100`), `fallback_score_hex=0x0.0p+0`, and status codes
`valid_finite=0`, `nonfinite_input_reject=1`. Before embedding or evaluating the
filter, one TensorFlow predicate checks whether every input coordinate is
finite. `tf.cond` returns the analytic finite target branch or the deterministic
reject pair. Batch evaluation applies the same per-row branch.

The frozen triggering inputs are scalar
`(NaN,-0.08,0.65,0.05)`, `(Inf,-0.08,0.65,0.05)`, and ordered batch rows
`(truth_free, NaN-row, Inf-row)`. Their JSON strings use respectively `nan`,
`inf`, and finite `float.hex()` strings. Tests require scalar status `1`, batch
statuses `(0,1,1)`, exact `-1e100`, exact zero score, and no filter invocation
for rejected rows through an injected test-only finite branch containing a
runtime `tf.debugging.assert_all_finite` guard. They also require
`GradientTape(log_prob)` equal to zero. The seam is dependency injection of the
internal finite-branch callable at construction; production defaults to the
real filter, the seam is excluded from signatures, and production rejects a
non-default callable unless an explicit `testing_only=True` constructor flag is
set. A target constructed with `testing_only=True` must expose capability
authority `debug_only`, `xla_hmc_ready=False`,
`full_chain_xla_diagnostic_ready=False`, and the testing scope/nonclaims frozen
in `golden-signatures.json`; its `target_signature()` and
`adapter_signature()` raise `RuntimeError`, and the A1 harness refuses to
serialize CPU/GPU evidence from it. A test-only target therefore cannot publish
production provenance or capability. Tests also require status `0` and bitwise
equality between wrapped and direct valid branches at both A0 anchors.

This is an input-domain reject convention, not posterior mass evidence. It does
not catch exceptions inside XLA, use `tf.py_function`, or reclassify a finite
filter failure.

### Authority Boundary

The source declares analytic authority `graph_native`, default compile mode
`xla`, `xla_hmc_ready=False`, and `full_chain_xla_diagnostic_ready=False`.
Neither readiness field is mutated by A1. The A1 result may report only
`GPU_XLA_CANARY_PASSED_AT_10_FROZEN_POINTS`; it cannot promote target-wide or
full-chain GPU/XLA/HMC readiness. HMC authority belongs to A5.

## Predeclared Test Points And Tolerances

The test point set is frozen before implementation:

- `truth_free` from A0;
- `phase2s_center = (0.5704394246369003,-0.1242247342531544,
  0.6609123192759063,0.1354211218811133)`;
- eight shell points `phase2s_center +/- 0.25 * scale_i * e_i`, with
  `scale=(0.35,0.35,0.35,0.35)` and coordinate order fixed by the mask.

The shell radius `0.25` is an inherited local diagnostic convenience, not a
scientific or posterior-coverage threshold. A branch failure triggers repair;
points may not be deleted after results are seen.

| Check | Exact rule and provenance | Role |
| --- | --- | --- |
| Historical route value and score at all ten finite points | The harness first verifies exact file SHA-256 `fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28`, then loads `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py` through `importlib.util.spec_from_file_location`, constructs `build_filtering_geometry_target()`, and calls its eager `_value_and_score_impl` at every frozen point. For historical scalar `h` and A1 scalar `a`, pass iff both are finite and `abs(a-h) <= 8*(2**-52)*max(1,abs(a),abs(h))`. For historical score `h[4]` and A1 score `a[4]`, pass iff every coordinate is finite and `max_i(abs(a_i-h_i)) <= 8*(2**-52)*max(1,max_i(abs(a_i)),max_i(abs(h_i)))`. These formulas cover zero/near-zero through the literal scale floor `1`. Truth and center additionally match the A0 probe payload under the same formulas. The production module never imports this route. | Estimand-preservation veto |
| Score finite difference | Centered step `1e-5`; `rtol=5e-3`, `atol=8e-4`, inherited from `tests/test_ssl_lstm_sgqf_ukf_adapters.py` SVD-UKF admission test | Derivative-admission veto only |
| Eager versus CPU-XLA value | Symmetric scale-aware bound `abs(a-b) <= 1e-10 * max(1,abs(a),abs(b))`; see the pre-evidence empirical amendment below | Engineering veto |
| Eager versus CPU-XLA score | `atol=1e-8`, `rtol=0`, inherited from nonlinear value/score chain parity tests | Engineering veto |
| CPU-XLA versus GPU-XLA | Same symmetric scale-aware value formula and `1e-8` absolute score bound, plus exact target/mask signatures | Target-only GPU/XLA engineering veto |
| Embedding/extraction | Exact tensor equality and inverse roundtrip | Engineering veto |

Finite-difference results do not establish posterior correctness. A tolerance
failure may indicate analytic derivative, branch, finite-difference step, or
harness failure; preserve diagnostics and discriminate before changing a bound.

### Pre-Evidence CPU-XLA Value-Parity Amendment

The original absolute `1e-10` value bound was inherited before this target's
XLA path was observed. During implementation localization, before any CPU or
GPU evidence artifact was opened, the unchanged ten-point set produced eager
versus CPU-XLA value residuals from `2.3845814212108962e-11` through
`1.9697381503647193e-09`. Nine of ten points exceeded the absolute bound. The
score residual maximum was `5.6968261219481064e-09`, within its unchanged
absolute `1e-8` bound. Disabling CPU XLA fast math did not change any residual.
At the worst value point, the final filtered mean and covariance residuals were
`3.469446951953614e-18` and `8.978549240895584e-20`; the mismatch was isolated
to accumulated log-likelihood arithmetic rather than target, state recursion,
filter branch, or wrapper drift.

The repaired value rule is therefore exactly
`abs(a-b) <= 1e-10 * max(1,abs(a),abs(b))` for both eager/CPU-XLA and
CPU-XLA/GPU-XLA comparisons. It is symmetric, preserves an absolute `1e-10`
floor near zero, scales with the compared log-target magnitude, and is not set
from the observed maximum. `value_parity_atol_hex` in both artifact designs is
the Python `float.hex()` string for `1e-10` and denotes the coefficient in this
literal max-scale formula, not an absolute-only `isclose` tolerance. The score
bound, historical `8*eps64` formulas, finite-difference bounds, frozen points,
target, signatures, and all nonclaims are unchanged.

This amendment is a numerical engineering contract only. It does not promote
posterior correctness, HMC/NeuTra readiness, predictive equivalence,
calibration, model adequacy, performance, or default/product readiness.

## Required Checks And Execution Order

### Artifact Schemas And Aggregation

`cpu-reference.json` has schema
`bayesfilter.ssl_lstm_completion.phase_a1_cpu_reference.v1` and
`gpu-xla-canary.json` has schema
`bayesfilter.ssl_lstm_completion.phase_a1_gpu_xla_canary.v1`. Each rejects
duplicate keys/nonfinite JSON and has exactly these top-level keys:

```text
schema_version, artifact_role, status, created_at_utc, run_manifest,
a0_bindings, a1_signatures, boundary_bindings, source_files,
test_point_design, point_results, reject_results, contract_checks,
evidence_signature, nonclaims
```

The CPU artifact has exact `artifact_role="phase_a1_cpu_hidden_reference"` and
`status="phase_a1_cpu_reference_passed"`; the GPU artifact has exact
`artifact_role="phase_a1_trusted_gpu_xla_canary"` and
`status="phase_a1_gpu_xla_canary_passed"`. `created_at_utc`,
`started_at_utc`, and `completed_at_utc` are UTC RFC 3339 strings with an
explicit `+00:00` suffix. `evidence_signature` is a lowercase 64-character
SHA-256 string.

Exact nested contracts:

- Unless a field below is explicitly an array/object/boolean/number, it is a
  JSON string. Every `_sha256` value is a lowercase 64-character hexadecimal
  string. Every `_hex` scalar is a Python `float.hex()` string; every score/input
  hex value is a fixed-length array of those strings. Every field ending in
  `_passed`, every row `passed`, and every `contract_checks` member except the
  explicitly typed `cpu_reference_file_sha256` is a JSON boolean. `status`
  fields in point/reject rows are JSON integers.
- `a0_bindings` has exactly `{target_semantic_sha256,
  signature_aggregate_sha256,immutable_aggregate_sha256,
  target_lock_file_sha256,dependency_manifest_file_sha256,
  observation_raw_sha256,full_fixture_raw_sha256}` with the literal values bound
  by A0 and `golden-signatures.json`;
- `a1_signatures` has exactly `{target_semantic_sha256,
  parameter_mask_sha256,masked_posterior_contract_sha256,
  golden_signatures_file_sha256}` with the four literal values
  `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`,
  `9dc25c878760b2fec5b5ad223662912272c2bda1b0d31590e3f60ec11ef79043`,
  `004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556`,
  and `04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34`;
- `boundary_bindings` has exactly `{a0_entry_verification_file_sha256,
  pre_run_scoped_boundary_file_sha256,
  protected_dependency_aggregate_sha256,
  excluded_dependency_aggregate_sha256,
  scoped_boundary_aggregate_sha256}`. CPU computes all five from the two strict
  pre-run artifacts; GPU copies them byte-for-byte from the accepted CPU
  artifact. Every CPU generation/verification and GPU generation/verification
  strictly loads both artifacts, rejects duplicate/extra/missing keys, repeats
  the safe anchor-to-current commit check, rehashes every protected dependency
  row, verifies that protected plus excluded rows exhaust the A0 manifest,
  recomputes both partition aggregates, validates every immutable input and the
  literal `owned_exact` path set, and recomputes the scoped artifact aggregate
  from its stored projection. The frozen entry's creation commit/history fields
  are validated as canonical creation-time provenance and are not compared with
  the later process commit/history. `initial_owned_state` is likewise validated
  as canonical creation-time provenance inside that stored projection; it is not compared
  with current A1-owned bytes, kinds, modes, or existence after the boundary is
  created. Current A1-owned source/evidence identity is instead carried by
  `source_files`, artifact-file hashes, focused checks, and the hash-pinned
  implementation/result reviews. This deliberately permits the supervised
  phase steps to create or repair the declared harness, CPU/GPU artifacts,
  result, governance records, and A2 handoff without weakening the
  protected/immutable checks. It does
  not compare the current unrelated Git index/porcelain digest to the stored
  explanatory snapshot. This is the operational definition of
  `contract_checks.scoped_boundary_verified`. It also reconstructs the exact
  v2 `entry_documents` projection from current files/reviews, requires the
  `target_lock_integrity` object and protected rows to match the preflight
  contract, and rejects any extra/missing nested entry keys. It opens the entry
  artifact read-only and never invokes its writer. This is the
  operational definition of `contract_checks.a0_entry_verified`;
- `source_files`: lexicographically sorted rows `{path,sha256,git_status}` for
  the production module, lazy export file, test, harness, golden-signature file,
  and historical comparator. `git_status` is `"clean"` for a clean tracked
  file, `"!!"` for an ignored file, or the exact two-character porcelain v1 XY
  field otherwise (`"??"` for untracked). The harness obtains ignored status
  with scoped `git status --porcelain=v1 --ignored -- <path>` after ordinary
  scoped porcelain returns no row; paths use repository-relative POSIX text;
- `test_point_design` is an object with exactly `{finite_points,
  nonfinite_cases}`. Its ten `finite_points` rows have exactly
  `{name,role,input_hex,finite_difference_step_hex,fd_rtol_hex,fd_atol_hex,
  historical_rtol_hex,historical_atol_hex,value_parity_atol_hex,
  score_parity_atol_hex}`. Its three `nonfinite_cases` rows have exactly
  `{name,role,input_strings}`. Finite order is exactly
  `truth_free`, `phase2s_center`, then for coordinate `0..3` its `minus` and
  `plus` shell rows. Nonfinite case order is `nan_scalar`, `inf_scalar`,
  `truth_nan_inf_batch`. `historical_rtol_hex` is the Python hex encoding of
  `8*(2**-52)`; `historical_atol_hex` is zero. These two stored fields describe
  the literal max-scale formulas above and must not be applied as library
  `isclose` parameters with different scaling;
- CPU `point_results`: the same ten finite rows with exactly
  `{name,input_hex,historical_value_hex,historical_score_hex,eager_value_hex,
  eager_score_hex,cpu_xla_value_hex,cpu_xla_score_hex,status,
  finite_difference_score_hex,historical_value_abs_residual,
  historical_score_abs_residual_inf,value_abs_residual,
  score_abs_residual_inf,fd_abs_residual_inf,fd_relative_residual_inf,
  historical_value_tolerance,historical_score_tolerance,value_tolerance,
  score_tolerance,fd_atol,fd_rtol,passed}`. GPU rows use exactly
  `{name,input_hex,cpu_xla_value_hex,cpu_xla_score_hex,gpu_xla_value_hex,
  gpu_xla_score_hex,status,value_abs_residual,score_abs_residual_inf,
  value_tolerance,score_tolerance,passed}`. Residual/tolerance fields are finite
  decimal JSON numbers; target values/scores are exact hex strings;
- CPU `reject_results` has exactly three rows in the frozen case order with
  `{name,input_strings,value_hex,score_hex,status,gradient_hex,
  finite_branch_runtime_assertion_not_triggered,passed}`. Scalar rows encode
  scalar value/status and four-coordinate score/gradient; the batch row encodes
  three values/statuses and `[3,4]` score/gradient. GPU `reject_results` is
  exactly `[]`; reject evidence is CPU contract evidence, not part of the
  ten-point GPU canary;
- CPU `contract_checks` has exactly
  `{a0_entry_verified,scoped_boundary_verified,mask_schema_valid,
  mask_golden_digest_match,wrapper_schema_valid,wrapper_golden_digest_match,
  golden_file_exact,historical_source_hash_exact,
  historical_all_ten_points_passed,target_semantic_digest_match,
  embed_extract_exact,prior_convention_exact,
  scalar_shapes_exact,batch_shapes_exact,batch_sizes_1_4_10_exact,
  callable_aliases_exact,compiled_default_invoked,valid_branch_bitwise_equal,
  nonfinite_input_reject_exact,reject_gradient_zero,finite_filter_failure_loud,
  testing_only_provenance_unavailable,testing_only_artifact_refused,
  finite_difference_passed,eager_cpu_xla_passed,no_benchmark_import,
  no_numpy_algorithmic_path,no_tf_py_function,historical_filter_route_only,
  authority_not_self_certified,point_order_exact,all_passed}`;
- GPU `contract_checks` has exactly
  `{cpu_reference_file_sha256,cpu_reference_verified,gpu_visible,
  trusted_provenance_recorded,jit_compile_true,xla_executed,
  gpu_device_placement_verified,tf32_recorded,cpu_gpu_parity_passed,
  signatures_equal,point_order_exact,all_passed}`;
- `nonclaims`: exactly the ordered seven-string array under
  `masked_posterior_contract.payload.nonclaims` in the immutable
  `golden-signatures.json`; no other prose nonclaim list is an artifact source.

`run_manifest` has exactly: `git_commit`, `git_dirty`, `command`, `cwd`,
`interpreter`, `python_version`, `packages`, `conda_env`, `environment`,
`physical_devices`, `logical_devices`, `cpu_gpu_status`, `trust_basis`, `dtype`,
`jit_compile`, `xla`, `tf32_enabled`, `data_version`, `random_seeds`,
`started_at_utc`, `completed_at_utc`, `wall_time_seconds`, `output_path`,
`log_path`, `plan_path`, and `result_path`. `random_seeds` is the string
`N/A_deterministic_target_no_randomness`; no numeric seed is invented. Types
and nested fields are binding:

- `git_commit`, `command`, `cwd`, `interpreter`, `python_version`, `conda_env`,
  `cpu_gpu_status`, `trust_basis`, `dtype`, `data_version`, `random_seeds`, all
  timestamp/path fields, and `output_path`/`log_path` are strings;
- `git_dirty`, `jit_compile`, `xla`, and `tf32_enabled` are booleans;
  `wall_time_seconds` is a finite nonnegative JSON number;
- `packages` has exactly string fields `{tensorflow,
  tensorflow_probability_distribution,numpy}`;
- `environment` has exactly string fields `{CUDA_VISIBLE_DEVICES,
  PYTHONHASHSEED,TF_DETERMINISTIC_OPS,TF_ENABLE_ONEDNN_OPTS,
  TF_NUM_INTRAOP_THREADS,TF_NUM_INTEROP_THREADS,OMP_NUM_THREADS,
  TF_CPP_MIN_LOG_LEVEL}`. GPU `CUDA_VISIBLE_DEVICES` is the literal
  `"not_set"`; CPU is `"-1"`. GPU thread variables not set by its exact command
  are literal `"not_set"`;
- `physical_devices` and `logical_devices` are lexicographically sorted arrays
  of exactly `{device_type,name}` string objects constructed from
  `tf.config.list_physical_devices()` and `tf.config.list_logical_devices()` by
  mapping TensorFlow device `.device_type` and `.name`, then sorting by
  `(device_type,name)`. CPU may have CPU rows but no GPU rows, while GPU must
  have at least one row with `device_type="GPU"` in each array.

Exact non-time manifest values are:

- both artifacts: `git_commit` is the actual process `HEAD`, captured before
  TensorFlow work and required to remain identical after all recomputation and
  before artifact publication. It need not equal the scoped boundary's earlier
  `boundary_creation_commit`: the immutable A0 anchor
  `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` must be an ancestor of both, and
  commit-history path enumeration from the anchor through each commit must be
  disjoint from the protected and A1-owned sets. CPU and GPU artifact commits
  may differ only under that safe rule. Verification checks the recorded run
  commit and the current verifier commit independently; it does not require
  them to be equal. `git_dirty=true`,
  `cwd="/home/ubuntu/python/BayesFilter"`, interpreter and versions are those
  fixed below, `conda_env="tfgpu"`, `dtype="float64"`,
  `data_version="aeb9a5e4b8cfe1ce374f66d5e145f8e5fb46e8d4a6586e62d573ebba3dc10f98"`,
  `random_seeds="N/A_deterministic_target_no_randomness"`,
  `jit_compile=true`, and `xla=true`;
- CPU: `cpu_gpu_status="cpu_hidden_no_gpu_visible"`,
  `trust_basis="cpu_hidden_reference_exception_not_gpu_evidence"`, environment
  values are exactly `CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`,
  `TF_DETERMINISTIC_OPS=1`, `TF_ENABLE_ONEDNN_OPTS=0`,
  `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`, `OMP_NUM_THREADS=1`,
  and `TF_CPP_MIN_LOG_LEVEL=1`;
- GPU: `cpu_gpu_status="trusted_gpu_visible_compiled_output_on_gpu"`,
  `trust_basis="owner_designated_managed_session_visible_gpu_trusted"`,
  environment values are exactly `CUDA_VISIBLE_DEVICES=not_set`,
  `PYTHONHASHSEED=0`, `TF_DETERMINISTIC_OPS=1`,
  `TF_ENABLE_ONEDNN_OPTS=0`, `TF_NUM_INTRAOP_THREADS=not_set`,
  `TF_NUM_INTEROP_THREADS=not_set`, `OMP_NUM_THREADS=not_set`, and
  `TF_CPP_MIN_LOG_LEVEL=1`;
- `tf32_enabled` is the JSON boolean returned by
  `tf.config.experimental.tensor_float_32_execution_enabled()` after TensorFlow
  import. The CPU/GPU verifier repeats that query in its own environment;
- `command` is the exact one-line shell-equivalent string formed by joining the
  displayed command's tokens with one ASCII space and no shell continuations;
  `output_path`, `log_path`, `plan_path`, and `result_path` are the exact
  repository-relative paths declared in this subplan. CPU and GPU harness
  constants own those strings and strict verification compares them literally.

The locked runtime is `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, Python
`3.13.13`, TensorFlow `2.20.0`, TensorFlow Probability distribution `0.25.0`,
and NumPy `2.1.3`. Any mismatch is a stop-and-plan-refresh condition; do not
install or substitute packages. CPU `physical_devices`/`logical_devices` must
record no GPU and trust basis `cpu_hidden_reference_exception_not_gpu_evidence`.
GPU status requires at least one physical and logical GPU, verified placement
of compiled output tensors on GPU, and trust basis exactly
`owner_designated_managed_session_visible_gpu_trusted` unless a later reviewed
human directive names another trust basis. `tf32_enabled` is recorded exactly
as reported; float64 parity, not TF32, is the target gate.

`evidence_signature` is SHA-256 over UTF-8 canonical JSON (`sort_keys=True`,
separators `,`/`:`, `ensure_ascii=True`, `allow_nan=False`) of exactly:

```text
{schema_version, artifact_role, a0_bindings, a1_signatures, boundary_bindings,
 source_files, test_point_design, point_results, reject_results,
 contract_checks, nonclaims}
```

It excludes timestamps, wall time, command, and device descriptions but does
not replace the exact artifact-file SHA-256 recorded in the result/review. The
harness implements `--verify` for each artifact, independently recomputes every
source/hash/signature/check/result and the evidence projection, and rejects
extra/missing keys. The GPU artifact additionally records and verifies the exact
CPU artifact file SHA-256 in `contract_checks.cpu_reference_file_sha256`.

### 1. Static And CPU-Hidden Checks

Create only the declared repository artifact directory and `/tmp` scratch:

```bash
mkdir -p docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1
mkdir -p /tmp/bayesfilter-ssl-lstm-a1-pycache
mkdir -p /tmp/bayesfilter-ssl-lstm-a1-runtime
```

Run exactly with the locked interpreter; all commands require exit `0`:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
bayesfilter/nonlinear/ssl_lstm_posterior_tf.py \
tests/test_ssl_lstm_posterior_tf.py \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py

CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache \
PYTEST_ADDOPTS='-p no:cacheprovider' \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
tests/test_ssl_lstm_posterior_tf.py \
tests/test_ssl_lstm_sgqf_ukf_adapters.py \
tests/test_ssl_lstm_protocol.py \
tests/test_nonlinear_ssm_phase3_value_score_chain.py

CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py \
--mode cpu-reference \
--output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json \
--log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.log

CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py \
--verify docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json
```

The harness commands emit respectively JSON `status` values
`phase_a1_cpu_reference_passed` and `phase_a1_artifact_verified`. Required test
coverage includes:

1. Mask validation, embed/extract roundtrip,
   signatures, prior convention, invalid shapes/dtypes/config, explicit
   finite-reject branch, valid-branch noninterference, exact scalar/batch order,
   hash-pinned historical-route replay at all ten finite points, finite
   differences at all ten fixed points, and
   eager/CPU-XLA parity.
2. Scoped whitespace and source audits proving no benchmark import,
   `tf.py_function`, NumPy algorithmic path, default `jit_compile=False`, or
   principal-square-root target migration.
3. Strict artifact verification and exact source-file hashes before review.

Run these read-only source audits; each requires exit `1` because no forbidden
match may be found:

```bash
rg -n 'docs\.benchmarks|benchmark_scalar_ssl_lstm|tf\.py_function|tf_principal_sqrt_ukf_score' \
bayesfilter/nonlinear/ssl_lstm_posterior_tf.py
rg -n 'import numpy|from numpy|jit_compile=False' \
bayesfilter/nonlinear/ssl_lstm_posterior_tf.py
```

Then run the exact non-vacuous whitespace audit below. Tracked paths use
`git diff --check`; untracked paths use `git diff --no-index --check`, whose
expected clean-file exit is `1` with empty output. Any other exit or any output
fails. Logs and structured JSON are validated by their strict parsers and are
not lexical-diff inputs.

```bash
bash -c '
set -u
paths=(
  bayesfilter/nonlinear/ssl_lstm_posterior_tf.py
  bayesfilter/nonlinear/__init__.py
  tests/test_ssl_lstm_posterior_tf.py
  docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py
  docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json
  docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md
)
for path in "${paths[@]}"; do
  test -f "$path" || { printf "missing required whitespace-audit path: %s\n" "$path"; exit 1; }
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    git diff --check -- "$path" || exit 1
  else
    output="$(git diff --no-index --check /dev/null "$path" 2>&1)"
    code=$?
    if test "$code" -ne 1 || test -n "$output"; then
      printf "%s\n" "$output"
      exit 1
    fi
  fi
done
'
```

This command must exit `0`. These lexical audits complement, but do not replace,
the runtime contract tests.

A CPU-hidden TensorFlow `cuInit` startup message is recorded only as a framework
anomaly. It is not GPU evidence or a machine diagnosis.

### 2. Read-Only Implementation Review

Use a fresh read-only reviewer distinct from the implementing Codex context.
Review the production module as one exact path, then
`bayesfilter/nonlinear/__init__.py`, the test, harness, and CPU artifact paths
one at a time as requested. The already accepted subplan review binds the
golden-signature file. The implementation review record binds exact SHA-256 for
the production module, lazy-export file, test, harness, golden-signature file,
historical comparator, and CPU artifact plus the CPU evidence signature.
Any mutation invalidates that verdict and requires CPU regeneration, focused
checks, and a fresh review. Do not start the GPU canary until the source and CPU
evidence receive bounded review agreement.

Claude remains policy-unavailable unless the governing boundary changes; use
fresh bounded `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude.

### 3. Trusted GPU/XLA Canary

Run only after the reviewed CPU gates pass. Exact command, requiring exit `0`:

```bash
env -u CUDA_VISIBLE_DEVICES -u TF_NUM_INTRAOP_THREADS \
-u TF_NUM_INTEROP_THREADS -u OMP_NUM_THREADS \
PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_CPP_MIN_LOG_LEVEL=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py \
--mode gpu-xla-canary \
--cpu-reference docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json \
--output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json \
--log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.log

env -u CUDA_VISIBLE_DEVICES -u TF_NUM_INTRAOP_THREADS \
-u TF_NUM_INTEROP_THREADS -u OMP_NUM_THREADS \
PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_CPP_MIN_LOG_LEVEL=1 \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-ssl-lstm-a1-pycache \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py \
--verify docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json
```

The commands emit `phase_a1_gpu_xla_canary_passed` and
`phase_a1_artifact_verified`. They use the repository TensorFlow/TFP
GPU path with `jit_compile=True`, `float64`, and structured provenance. This is
a target-only value/score canary at the ten frozen points, not HMC, training, a
benchmark, or performance claim. The artifact must record:

- physical/logical GPU identity and visibility;
- owner-designated managed-session trust basis when applicable;
- XLA/JIT, TF32, dtype, TensorFlow/TFP/CUDA-visible settings;
- A0 target/immutable aggregates and A1 target/mask/adapter signatures;
- CPU-XLA comparator artifact hash;
- per-point value/score residuals and branch codes;
- command, commit, dirty status, conda environment, wall time, and outputs.

A nontrusted/sandbox GPU failure is not machine evidence; rerun the same command in the trusted
context. A trusted GPU/XLA compile or parity failure is an A1 repair trigger,
not permission for CPU-only production or `jit_compile=False` default.

## A1 Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does a production-owned four-coordinate TensorFlow target preserve the exact A0 historical estimand and expose a graph-native target-only GPU/XLA value/score surface? |
| Exact baseline | A0 target lock plus the hash-pinned historical target constructor evaluated eagerly at all ten frozen finite points; the historical benchmark remains test provenance only, not a production dependency |
| Primary pass criterion | Typed mask/config/target contracts pass; exact target signatures bind; all-ten-point historical route, prior, embedding, derivative, scalar/batch, eager/CPU-XLA gates pass; trusted GPU/XLA canary passes at exactly the ten frozen finite points |
| Promotion vetoes pending repair | Code/test/schema/signature defect, anchor/parity/finite-difference failure, invalid finite-reject behavior, source-review finding, or newly generated A1 structured artifact that is missing/corrupt |
| Immediate continuation vetoes | A0 immutable drift, target/filter/prior migration, inability to reproduce the locked target after verified harness repair, graph-native/XLA route impossible without forbidden bridge, or required edit outside reviewed scope |
| Explanatory only | Compile/runtime, trace counts, residuals within tolerance, branch telemetry, shell values, and device utilization |
| What will not be concluded | Posterior correctness, HMC/NeuTra readiness, convergence, predictive equivalence, model adequacy, superiority, performance, or default/public readiness |
| Preservation | A1 source/tests/harness, JSON/log/Markdown result, review records, and refreshed A2 subplan |

## Diagnostic Role Classification

| Diagnostic | Role |
| --- | --- |
| A0 target/mask/observation/prior signature match | Promotion criterion and continuation veto on unexplained drift |
| Embed/extract, shape/dtype, prior, invalid-input tests | Engineering promotion criterion |
| Hash-pinned historical route replay at all ten points | Estimand-preservation promotion criterion |
| Central finite difference | Derivative-admission promotion veto; not posterior evidence |
| Eager/CPU-XLA and CPU/GPU-XLA parity | Engineering promotion veto |
| Trusted GPU visibility and XLA compilation | Promotion criterion only for status `GPU_XLA_CANARY_PASSED_AT_10_FROZEN_POINTS`; no target-wide readiness claim |
| Runtime, compile time, shell values, residuals below bounds | Explanatory only |
| HMC, predictive, calibration, model-adequacy diagnostics | Not assessed in A1 |

## Skeptical Plan Audit

| Risk | A1 control |
| --- | --- |
| Wrong baseline | A0 target lock is the comparator; principal-square-root target and Phase 2V are excluded |
| Proxy promotion | Unit/parity tests can pass A1 engineering only, not HMC or scientific gates |
| Missing stop conditions | Immutable drift, target migration, forbidden bridge, trusted GPU failure after repair, and repair exhaustion stop handoff |
| Unfair comparison | CPU and GPU evaluate identical points, dtype, target, signatures, and XLA program |
| Hidden assumptions | Prior normalization, finite reject, mask dimension, score authority, points, and tolerances are explicit |
| Stale context | Exact A0 lock/review identities, protected target-critical hashes, and all-ten-point semantic replay are mandatory at entry and result; unrelated concurrent paths are explanatory only |
| Environment mismatch | CPU reference is deliberately hidden; GPU evidence requires trusted provenance |
| Artifact does not answer question | Structured per-point values/scores/signatures and source audits directly test estimand preservation |

Audit status: `PASSED_FOR_A1_MATERIAL_REVIEW_ONLY`. No A1 runtime is authorized
until the A0 final handoff and this review both pass.

## Repair And Review Bounds

An implementation repair cycle is one mutation of A1 source/test/harness or an
A1 structured artifact in response to one named failed gate, followed by the
focused failing check and every prerequisite check that mutation could affect.
Allow at most five cycles for the same normalized blocker class:
`code_or_test_contract`, `source_review`, `mask_or_signature`,
`target_value_or_score`, `historical_route_parity`, `finite_difference`,
`nonfinite_input_reject`, `callable_shape_or_batch`, `cpu_xla`, `gpu_xla`,
`artifact_schema_or_integrity`, or `environment_or_device_provenance`.
Every promotion-veto finding in the evidence contract is normalized to exactly
one of these classes before mutation; a finding spanning classes increments
each affected class. Renaming a blocker does not reset its count.
Changing a tolerance, frozen point, target setting, baseline, evidence role, or
schema is not an implementation repair; it requires stopping and reviewing a
plan amendment before any further result is opened. Five failed repair cycles
for one blocker produce a blocker result and stop.

Each material review target (subplan, golden-signature semantic contract,
implementation packet represented by its one-path reviews, A1 result, and A2
subplan) separately allows at most five
substantive `AGREE`/`REVISE` rounds for the same unresolved blocker. No-verdict
prompt/tool recovery is recorded but does not count. Every review uses a fresh
read-only reviewer context distinct from the implementing/supervising context
and records exact reviewed file SHA-256 values. Any reviewed-file mutation
invalidates the prior verdict and requires rereview.

Owner authorization on 2026-07-12 grants up to five additional substantive
review rounds only for normalized blocker
`entry_artifact_live_history_lifecycle`, beginning with recovery Round E1.
This exceptional budget does not erase the original five-round history, reset
any implementation-repair counter, authorize a tolerance/schema/target change,
or broaden runtime/scientific/product boundaries. Agreement closes the blocker
and returns to the ordinary plan; five additional `REVISE` verdicts close a new
blocker result and stop.

## Required A1 Result

The close record must contain decision and inference-status tables; separate
engineering, sampler, predictive-equivalence, calibration, and adequacy
ledgers; exact source and artifact hashes; CPU and GPU run manifests; all fixed
test points and residuals; finite-reject branches; review rounds/repairs;
candidate-versus-direction classification; post-run red team; explicit
nonclaims; and the exact A2 handoff.

## Forbidden Claims And Actions

- Do not import benchmark modules or hard-code an unverified copy without A0
  hash/signature tests.
- Do not change observations, free coordinates/order, fixed values, prior,
  historical SVD-UKF route, numerical settings, or dtype.
- Do not use NumPy, `tf.py_function`, JAX, or PyTorch in algorithmic/gradient
  paths.
- Do not run HMC, NeuTra, geometry fitting, forecasting, calibration, sweeps,
  or performance benchmarks.
- Do not claim full-chain HMC authority, posterior correctness, sampler
  validity, predictive equivalence, scientific validity, or default readiness.
- Do not stage, commit, push, reset, restore, clean, install packages, fetch
  network resources, or edit unrelated dirty work.

## Exact Next-Phase Handoff Conditions

All must pass:

1. Every required A1 source/test/harness path is inside the write set and passes
   scoped local checks and material review.
2. Production target/mask/adapter signatures bind A0 and all strict schema/hash
   checks pass.
3. Hash-pinned historical-route agreement at all ten finite points, finite
   differences, scalar/batch, eager/CPU-XLA, and
   trusted GPU/XLA gates pass at every frozen point.
4. Invalid-region behavior is graph-native, finite, deterministic, shaped, and
   proven not to affect valid inputs.
5. A1 result records both CPU and GPU manifests and is accepted by an
   independent, hash-pinned material review after all final reruns.
6. A2 subplan is drafted from actual A1 signatures, exact write set, terminal-
   state/filter parity contract, oracle prerequisites, and GPU/XLA boundaries,
   then independently and hash-pinned reviewed after the final A1 result is
   accepted.
7. No target, runtime, model-file, product/default, or scientific boundary is
   implicit.

Only then may A2 implementation begin.

## Stop Conditions

- A0 final handoff or immutable rehash is absent or changes.
- Target identity cannot be reproduced without benchmark imports or estimand
  migration.
- The graph-native historical score cannot satisfy XLA/parity after five
  reviewed repair cycles and no in-scope target-preserving route exists.
- A trusted GPU/XLA canary remains invalid after five repair cycles for the
  same normalized blocker.
- Predecessor/A0 evidence is missing/corrupt; newly generated A1 evidence remains
  missing/corrupt after five `artifact_schema_or_integrity` repair cycles;
  review does not converge within five substantive rounds for one blocker; or
  work requires an unapproved boundary.

An implementation candidate failure is not a research-direction rejection.
Record whether the defect lies in mask/config code, filter integration,
derivative path, finite reject, XLA compilation, environment, or evidence
artifact, and preserve the smallest next discriminating repair.

## Mandatory Phase-End Sequence

1. Verify the marked scoped concurrent-lane authorization in the approval
   ledger, final A0 entry identities, exhaustive protected/excluded manifest
   partition, and safe anchor-to-evidence commit relation; then record the A1
   pre-run evidence contract.
2. Implement only the reviewed write set and run the focused CPU-hidden checks.
3. Write and strictly verify the CPU artifact, then pass independent hash-pinned
   production/lazy-export/test/harness/CPU-artifact review.
4. Run and strictly verify the trusted GPU/XLA canary.
5. Perform the final evidence checkpoint without mutating reviewed evidence:
   strict-load and verify the frozen A0 entry artifact read-only, including its
   exact file hash and reconstructed v2 projections; do not rerun its writer.
   Independently capture current `HEAD`, repeat full anchor-to-current
   per-commit history enumeration, require no protected or A1-owned committed
   path and stable opening/closing `HEAD`, and record the final commit/path list
   for the result. Verify the scoped boundary by rehashing every protected row
   and immutable input, checking the owned-set literal, and recomputing the
   stored artifact aggregate without comparing current unrelated-worktree
   provenance; rerun the
   exact focused pytest command; run CPU `--verify` (which recomputes all CPU
   results); run GPU
   `--verify` (which recomputes all GPU results against the accepted CPU file);
   then record source and exact artifact hashes. If any source/test/harness/CPU
   artifact mutation is needed, return to step 3; if only the GPU artifact needs
   repair, return to step 4. Any mutation after the final checkpoint to
   source/test/harness, golden/entry/inventory, or CPU/GPU evidence invalidates
   it; the explicitly sequenced result/governance/A2 writes do not because they
   are excluded from evidence projections and the scoped boundary equality.
6. Write the A1 result from the final checkpoint and obtain independent
   hash-pinned review. Immediately after every result write/repair, require the
   exact result-only whitespace command below to exit `0`. Result-only repair
   requires rereview; it does not rerun step 5 because result/governance
   documents are excluded from evidence projections. Any evidence/source repair
   restarts step 5.
7. Draft A2 only after the final A1 result is accepted, then obtain its own
   independent hash-pinned review. A2-only repair requires A2 rereview. It may
   not change or reinterpret A1 evidence.
8. Advance only if every conjunctive condition passes; otherwise update the
   stop handoff with the exact blocker.

Exact result-only whitespace command for step 6:

```bash
bash -c '
path=docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md
test -f "$path" || { printf "missing required result: %s\n" "$path"; exit 1; }
if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
  git diff --check -- "$path"
else
  output="$(git diff --no-index --check /dev/null "$path" 2>&1)"
  code=$?
  test "$code" -eq 1 && test -z "$output" || { printf "%s\n" "$output"; exit 1; }
fi
'
```
