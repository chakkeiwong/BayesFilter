"""Evidence readers for actual-SV surrogate-HMC inference, without TF imports."""
import json
from pathlib import Path
from bayesfilter.inference.surrogate_hmc_program import file_hash, sv20_family_remaining_allocation

INPUT_NAMES = {"claim", "selection", "scope", "reference_report", "reference_manifest",
    "posterior_report", "posterior_manifest", "consumer_report", "consumer_manifest",
    "controls_report", "controls_manifest"}


def family_remaining(root):
    return sv20_family_remaining_allocation(dict(prior_runs=[dict(path=str(p))
        for p in Path(root).glob("docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json")]))


def sv20_endpoint_state(root):
    from scripts.qualify_surrogate_hmc_sv_endpoint import complete_forecast
    state = dict(qualified=None, status="NOT_RUN")
    paths = sorted(Path(root).glob("docs/plans/artifacts/surrogate-hmc-all-models-*-sv20-endpoint-*/report.json"))
    required = {"bitwise_replay", "canonical_call_chain", "claim_data_bound", "complete_transitions",
        "domain_mass", "endpoint_independent", "endpoint_sign", "field_sign_status", "force_healthy",
        "independent_typed_bindings", "permutation", "predictive_diagnostics", "proposal_reversible",
        "reference_force_sign", "reference_refinement", "row_independence", "unchanged_value_and_score"}
    for path in paths:
        report = json.loads(path.read_text())
        state["status"] = report["status"]
        if not report.get("qualified"):
            continue
        run = json.loads((path.parent/"run-manifest.json").read_text())
        accounting = json.loads((path.parent/"accounting.json").read_text())
        if (report["status"] != "QUALIFIED_INFERENCE_ENDPOINT" or report.get("retained_inference") is not False
                or not required.issubset(report["checks"]) or not all(report["checks"].values())
                or run.get("exit_code") != 0 or run.get("source_drift") or accounting.get("exit_code") != 0
                or set(run["inputs"]) != INPUT_NAMES or not run.get("source_hashes")):
            raise ValueError("invalid SV20 endpoint qualification")
        for name, digest in run["source_hashes"].items():
            if file_hash(path.parent/"source"/name) != digest:
                raise ValueError("archived SV20 endpoint source changed")
            if name.startswith("bayesfilter/") and name not in (
                    "bayesfilter/inference/surrogate_hmc_program.py", "bayesfilter/inference/__init__.py"):
                if file_hash(Path(root)/name) != digest:
                    raise ValueError("SV20 endpoint numerical source changed: "+name)
        for record in run["inputs"].values():
            if file_hash(Path(record["path"])) != record["sha256"]:
                raise ValueError("SV20 endpoint input changed")
        forecast = complete_forecast(report["complete_transition_costs"], report["predictive_costs"])
        if report["forecast"] != forecast:
            raise ValueError("SV20 complete cost mismatch")
        state["qualified"] = dict(report=str(path.resolve()), report_sha256=file_hash(path),
            role="inference_endpoint_preparation", authorizes_new_inference=False, complete_forecast=forecast)
    return state


def sv20_inference_history(root):
    """Report a preserved SV20 B/R attempt; never authorize another closed look."""
    state = dict(status="NOT_RUN", surrogate_hmc_tested=False,
        retained_surrogate_hmc_tested=False, authorizes_new_inference=False,
        comparisons=dict.fromkeys(("B/R", "C/D", "D/R", "C/R"), "NOT_RUN"))
    artifacts = Path(root)/"docs/plans/artifacts"
    candidates = sorted(set(artifacts.glob("surrogate-hmc-all-models-*-sv20-inference-*/run-manifest.json")) |
        set(artifacts.glob("surrogate-hmc-all-models-*/P3-SV20-inference/run-manifest.json")))
    manifests = []
    for path in candidates:
        if "-tests-" in str(path.relative_to(artifacts)):
            continue
        metadata = json.loads(path.read_text())
        # A master manifest can exist before its child starts. It is accounting
        # for orchestration, not evidence of a previous statistical attempt.
        if path.parent.name == "P3-SV20-inference" or any(
                Path(arg).name == "run_surrogate_hmc_sv.py" for arg in metadata.get("command", [])):
            manifests.append(path)
    if not manifests:
        return state
    state["attempt_manifests"] = [str(path.resolve()) for path in manifests]
    if len(manifests) != 1:
        state.update(status="RECONCILIATION_REQUIRED", error="Multiple SV20 inference attempts require reconciliation")
        return state
    directory = manifests[0].parent
    path = directory/"report.json"
    try:
        run = json.loads(manifests[0].read_text())
        report = json.loads(path.read_text())
        state.update(report=str(path.resolve()), report_sha256=file_hash(path))
        if run.get("scope") != "SV20-T20-N1024" or report.get("model_id") != "SV20-T20":
            raise ValueError("wrong SV20 inference scope")
        if "exit_code" not in run:
            state.update(status="RUNNING", closed=False)
            return state
        if run["exit_code"] != 0 or report.get("status") != "COMPLETED_BOUNDED_INFERENCE":
            state.update(status=report.get("status", "INCOMPLETE_ATTEMPT"), closed=True,
                next_action="Inspect preserved SV20 attempt before a budgeted repair; do not repeat closed looks")
            return state
        accounting = json.loads((directory/"accounting.json").read_text())
        if accounting.get("exit_code") != 0 or not run.get("source_hashes") or run.get("source_drift"):
            raise ValueError("SV20 inference lacks successful accounting or source provenance")
        # Later reporting changes must not invalidate an otherwise preserved run.
        for name, digest in run["source_hashes"].items():
            if file_hash(directory/"source"/name) != digest:
                raise ValueError("archived SV20 inference source changed: "+name)
        if file_hash(directory/"data.json") != run["data_sha256"]:
            raise ValueError("SV20 inference data changed")
        if set(run["inputs"]) != INPUT_NAMES | {"endpoint_report", "endpoint_manifest"}:
            raise ValueError("missing frozen SV20 inference inputs")
        for record in run["inputs"].values():
            if file_hash(Path(record["path"])) != record["sha256"]:
                raise ValueError("frozen SV20 inference input changed")
        if json.loads((directory/"data.json").read_text()) != json.loads(Path(run["inputs"]["claim"]["path"]).read_text()):
            raise ValueError("SV20 retained data differs from qualified claim")
        if run.get("retained_looks") != [1000, 2000] or run.get("function_count") != 13 or run.get("comparison_families") != 4:
            raise ValueError("SV20 inference evidence contract changed")
        row = report["arms"]["B"]
        chunks, looks = row["chunks"], row["looks"]
        if [look["draws_per_chain"] for look in looks] not in ([], [1000], [1000, 2000]):
            raise ValueError("undeclared or repeated SV20 statistical look")
        if any(look["status"] not in ("INCONCLUSIVE", "EQUIVALENT", "DISCREPANCY") or
               (look["status"] != "INCONCLUSIVE" and look.get("diagnostic_valid") is not True) for look in looks):
            raise ValueError("invalid SV20 uncertainty conclusion")
        if any(look["status"] != "INCONCLUSIVE" for look in looks[:-1]):
            raise ValueError("SV20 continued after a closed statistical look")
        if row["tuning"]["passed"]:
            tune_path = Path(row["tuning"]["manifest"])
            tune = json.loads(tune_path.read_text())
            binding = json.loads((directory/"binding.json").read_text())
            if (tune.get("schema") != "bayesfilter.verified_tensorflow_hmc.v1" or
                    tune.get("artifact_authority") is not True or tune.get("handoff_eligible") is not True or
                    tune.get("passed") is not True or not tune.get("checks") or not all(tune["checks"].values()) or
                    tune["runner_binding"] != binding or binding["target_scope"] != "SV20-T20-N1024-claim-attempt1:B"):
                raise ValueError("SV20 public tuning or binding mismatch")
            if file_hash(tune_path.parent/"nomination/manifest.json") != tune["nomination_manifest_sha256"]:
                raise ValueError("SV20 nomination archive changed")
            for phase in ("warmup", "verification"):
                for record in tune[phase].values():
                    if file_hash(tune_path.parent/record["path"]) != record["sha256"]:
                        raise ValueError("SV20 tuning trace changed")
            for index, chunk in enumerate(chunks):
                archive_path = Path(chunk["manifest"])
                archive = json.loads(archive_path.read_text())
                previous = chunks[index-1]["manifest"] if index else None
                if (archive["tuning_manifest_sha256"] != file_hash(tune_path) or
                        archive["binding_hash"] != binding["binding_hash"] or
                        archive["health_passed"] != chunk["health_passed"] or
                        archive["health_passed"] != (bool(archive["checks"]) and all(archive["checks"].values())) or
                        archive["continuation_manifest"] != previous or
                        archive["config"]["continuation_manifest"] != previous or
                        archive["config"]["num_results"] != 250 or
                        archive["config"]["seed"] != [20260922, 36000+index]):
                    raise ValueError("SV20 retained archive, seed or continuation mismatch")
                for record in [archive["raw_samples"], archive["initial_state"], *archive["chunk"].values()]:
                    if file_hash(archive_path.parent/record["path"]) != record["sha256"]:
                        raise ValueError("SV20 retained sample or trace changed")
            if row["status"] == "HEALTH_VETO":
                if (not chunks or chunks[-1]["health_passed"] or any(not c["health_passed"] for c in chunks[:-1]) or
                        (looks and looks[-1]["status"] != "INCONCLUSIVE")):
                    raise ValueError("SV20 health veto lacks a terminal failed chunk")
            elif (not looks or not chunks or any(not c["health_passed"] for c in chunks) or
                    row["retained_draws"] != 250*len(chunks) or looks[-1]["draws_per_chain"] != row["retained_draws"] or
                    row["status"] != looks[-1]["status"] or
                    (row["status"] == "INCONCLUSIVE" and row["retained_draws"] != 2000)):
                raise ValueError("SV20 retained result is not a closed declared look")
        elif row["status"] != "TUNING_REJECTED" or chunks or looks:
            raise ValueError("SV20 rejected tuning contains retained draws")
        expected = dict(state["comparisons"], **{"B/R": row["status"]})
        if report["comparisons"] != expected:
            raise ValueError("SV20 result/comparison mismatch")
        state.update(status="COMPLETED_BOUNDED_INFERENCE", closed=True, artifact_valid=True,
            comparisons=expected, surrogate_hmc_tested=True, retained_surrogate_hmc_tested=bool(chunks),
            arms={"B": dict(status=row["status"], retained_draws=row.get("retained_draws", 0),
                tuning_passed=row["tuning"]["passed"], diagnostic_valid=looks[-1]["diagnostic_valid"] if looks else None)},
            run_manifest=str(manifests[0].resolve()),
            next_action="Preserve closed B/R looks; C/D needs an independently qualified finite-target reference")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(status="ARTIFACT_INVALID", artifact_valid=False, error=str(exc))
    return state

