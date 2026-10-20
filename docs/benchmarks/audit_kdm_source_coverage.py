"""CPU-only audit of saved KDM replication evidence; no algorithm or training execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
PLAN = "docs/plans/controlled-smc-kdm-source-closure-plan-20261014.md"
RESULT = "docs/plans/controlled-smc-kdm-source-closure-result-20261014.md"
CORE = ["bayesfilter/replication/" + name for name in (
    "kdm_iwsg_tf.py", "kdm_sampling_tf.py", "kdm_normal_von_mises_tf.py",
    "kdm_neural_replay_tf.py", "kdm_source_stages_tf.py",
    "kdm_training_controller.py", "kdm_stage_transfer_tf.py", "kdm_sequential_tf.py")]
REVIEWED_CHANGES = {
    "docs/benchmarks/diagnose_kdm_fresh_pretraining.py": (
        "91a089816730f95fccad0b526bf2e38c9e97c968a859b18bac1aa0d7626614f4",
        "Adds a cloud_law observation branch; existing update/evaluate/replay branch unchanged."),
    "docs/benchmarks/diagnose_kdm_native_evaluation_tf.py": (
        "0945e6412059cbcc4a4116d98313a89de70d8d93b31320a74120e0fc3c2ec103",
        "Replaces two input-scope assertions with exact checkpoint-set validation; numerical main body unchanged."),
    "docs/benchmarks/reference_kdm_recent_endpoint.py": (
        "d40a339e3a8e884056445fb82a0c9538b4d83de905a7f7e00b7d5470a151f6fa",
        "Adds optional unit02 MDPF intervention export; default recent-checkpoint export preserved.")}
PAPER = ".localresources/papers/younis-sudderth-2023-long-range-tracking.pdf"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git(directory, *args, binary=False):
    value = subprocess.run(["git", "-C", str(directory), *args], check=True, capture_output=True).stdout
    return value if binary else value.decode().strip()

def passing(data, key="checks", flag="passed"):
    rows = data.get(key)
    return isinstance(rows, list) and bool(rows) and all(
        isinstance(row, dict) and row.get(flag) is True for row in rows)

def node(path, owner, name):
    body = ast.parse(Path(path).read_text()).body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)

def calls(fun):
    return {ast.unparse(n.func) for n in ast.walk(fun) if isinstance(n, ast.Call)}

def references(fun):
    return {ast.unparse(n) for n in ast.walk(fun) if isinstance(n, (ast.Name, ast.Attribute))}

def binding(fun, target, implementation):
    dictionaries = {}
    for item in fun.body:
        if (isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)
                and ast.unparse(item.value.func) == "dict"):
            for dest in item.targets:
                dictionaries[ast.unparse(dest)] = {k.arg for k in item.value.keywords if k.arg}
    for item in ast.walk(fun):
        if not (isinstance(item, ast.Assign) and any(ast.unparse(t) == target for t in item.targets)
                and isinstance(item.value, ast.Call) and ast.unparse(item.value.func) == "tf.function"
                and item.value.args and ast.unparse(item.value.args[0]) == implementation):
            continue
        keys = {k.arg for k in item.value.keywords if k.arg}
        for kw in item.value.keywords:
            if kw.arg is None:
                keys |= dictionaries.get(ast.unparse(kw.value), set())
        if {"input_signature", "jit_compile"} <= keys:
            return True
    return False

def accepted(audit):
    return (bool(audit.rows) and bool(audit.checks)
            and all(row["status"] == "PASS" for row in audit.rows)
            and all(check["passed"] is True for check in audit.checks))

def self_test():
    from tempfile import TemporaryDirectory
    count = 0
    assert passing({"checks": [{"passed": True}]})
    count += 1
    for bad in ({}, {"checks": []}, {"checks": [{}]}, {"checks": [{"passed": False}]}):
        assert not passing(bad)
        count += 1
    f = ast.parse("def f():\n options = dict(input_signature=s, jit_compile=True)\n x = tf.function(self.real, **options)\n").body[0]
    assert binding(f, "x", "self.real") and not binding(f, "x", "self.wrong")
    count += 1
    missing = ast.parse("def f():\n x = tf.function(self.real)\n").body[0]
    assert not binding(missing, "x", "self.real")
    count += 1
    a = Audit()
    assert not accepted(a)
    count += 1
    a.group("good", "fixture", lambda: a.check("good", True))
    assert accepted(a)
    count += 1
    a.group("bad", "fixture", lambda: a.check("bad", False))
    assert not accepted(a)
    count += 1
    a = Audit()
    a.group("missing assertions", "fixture", lambda: None)
    assert not accepted(a)
    count += 1
    a = Audit()
    a.group("missing evidence", "fixture", lambda: a.read(Path("/this-file-does-not-exist.json")))
    assert not accepted(a)
    count += 1
    with TemporaryDirectory(prefix="kdm-audit-test-") as directory:
        path = Path(directory) / "source.py"
        path.write_text("pass\n")
        expected = sha(path)
        path.write_text("changed\n")
        a = Audit()
        a.group("hash changed", "fixture", lambda: a.file_hashes({str(path): expected}, "hash"))
        assert not accepted(a)
        count += 1
    a = Audit()
    a.group("broken call", "fixture", lambda: a.check("call", "missing.call" in calls(f)))
    assert not accepted(a)
    count += 1
    return {"status": "PASS", "cases": count,
            "coverage": ["missing/empty/false assertions", "changed source", "wrong compiled binding",
                         "absent call", "missing evidence", "failed verdict dependency"]}

class Audit:
    def __init__(self):
        self.checks, self.evidence, self.rows, self.drift = [], {}, [], []
    def read(self, path):
        path = Path(path)
        self.evidence[str(path)] = sha(path)
        return json.loads(path.read_text())
    def check(self, name, condition, detail=None):
        self.checks.append(dict(name=name, passed=bool(condition), detail=detail))
    def group(self, operation, anchor, fn):
        first = len(self.checks)
        try:
            fn()
        except Exception as exc:
            self.check(operation + "/read_or_schema", False, repr(exc))
        items = self.checks[first:]
        okay = bool(items) and all(c["passed"] for c in items)
        self.rows.append(dict(operation=operation, status="PASS" if okay else "BLOCKED",
                              source_anchor=anchor, check_names=[c["name"] for c in items]))
    def file_hashes(self, hashes, label, snapshot_root=None, allow_reviewed=False):
        for rel, expected in hashes.items():
            path = ROOT / rel
            actual = sha(path) if path.is_file() else None
            okay = actual == expected
            if not okay and allow_reviewed and rel in REVIEWED_CHANGES:
                frozen, reason = REVIEWED_CHANGES[rel]
                snapshot = Path(snapshot_root) / rel
                okay = actual == frozen and snapshot.is_file() and sha(snapshot) == expected
                self.drift.append(dict(path=rel, old_sha256=expected, current_sha256=actual,
                                       snapshot=str(snapshot), reviewed_reason=reason, accepted=okay))
            self.check(label + "/" + rel, okay, dict(expected=expected, actual=actual))
    def author_hashes(self, hashes, label):
        self.check(label + "/nonempty", bool(hashes))
        for rel, expected in hashes.items():
            blob = git(AUTHOR, "show", PIN + ":" + rel, binary=True)
            self.check(label + "/" + rel,
                       sha(AUTHOR / rel) == expected and hashlib.sha256(blob).hexdigest() == expected)

def run(args):
    started, cpu_start = time.monotonic(), time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    rd, ck = audit.read, audit.check
    verify = BASE / "stage-isolation-20261013/attempt03/verification"
    vm = rd(verify / "manifest.json")
    v = rd(verify / "diagnostic/result.json")
    source = rd(Path(vm["reference"]))
    baseline = rd(Path(source["baseline"]))
    def provenance():
        ck("author pin", git(AUTHOR, "rev-parse", "HEAD") == PIN)
        ck("paper PDF", sha(ROOT / PAPER) == "e380be4b87bbf33707fe599c2617c331c1881f95a073427aead25513c9a84d8c")
        ck("source reference pin", source["author_commit"] == baseline["author_commit"] == PIN)
        audit.author_hashes(source["author_hashes"], "author/stages")
        audit.author_hashes(baseline["author_hashes"], "author/full_sequence")
        ck("reference binding", sha(Path(vm["reference"])) == vm["reference_sha256"] == v["reference_sha256"])
        ck("verification result binding", sha(verify / "diagnostic/result.json") == vm["result_sha256"])
        ck("baseline binding", sha(Path(source["baseline"])) == source["baseline_sha256"])
        required = CORE + ["docs/benchmarks/diagnose_kdm_source_stages_tf.py",
                          "docs/benchmarks/diagnose_kdm_source_epoch_tf.py",
                          "docs/benchmarks/reference_kdm_neural.py",
                          "docs/benchmarks/reference_kdm_source_stages.py"]
        ck("required core hash coverage", all(p in vm["hashes"] for p in required))
        audit.file_hashes({p: vm["hashes"][p] for p in required}, "current stage runtime")
        ck("verification explicitly passes", v["status"] == "PASS" and passing(v) and
           v["checks_failed"] == 0 and v["checks_total"] == len(v["checks"]) == 2204)
        ck("verification GPU/XLA", v["jit_compile"] is True and v["tf32"] is False and
           v["memory_policy"]["all_physical_devices_memory_growth"] is True)
    audit.group("Source identity and current numerical implementation",
                "Author commit " + PIN + "; paper PDF; stage verification manifest", provenance)

    def stages():
        for split, expected in [("training", 5000), ("validation", 1000)]:
            d = source["datasets"][split]
            ck("data/" + split, d["rows"] == expected and sha(Path(d["path"])) == d["sha256"])
        ck("full variants and precision", {(c["variant"], c["dtype"]) for c in baseline["cases"]} ==
           {(x, y) for x in ("mdpf", "am_dpf") for y in ("float32", "float64")})
        for prefix in ["initial/" + variant + "/" + precision + "/" for variant in ("mdpf", "am_dpf")
                       for precision in ("float32", "float64")]:
            entries = [c for c in v["checks"] if c["name"].startswith(prefix)]
            ck(prefix, len(entries) == 12 and all(c["passed"] is True for c in entries))
        for precision in ("float32", "float64"):
            for prefix, minimum in [("pretrain/", 280), ("full/mdpf/", 310),
                                    ("full/am_dpf/", 380), ("transfer/", 80)]:
                p = prefix + precision + "/"
                entries = [c for c in v["checks"] if c["name"].startswith(p)]
                ck(p, len(entries) >= minimum and all(c["passed"] is True for c in entries))
        ck("transfer flag", v["stage_transfers_checked"] is True)
        cfg = source["full_training_config"]
        ck("released full training constants", cfg["number_of_particles"] == 25 and
           cfg["training_batch_size"] == 64 and cfg["validation_batch_size"] == 256 and
           cfg["truncated_bptt_modulo"] == 4 and cfg["gradient_clip_value"] == 100)
        ck("pretraining weight decay", source["pretraining_config"]["weight_decay"] == .001)
    audit.group("Data masks, architecture, initialization, heads, loss, TBPTT, Adam and stage transfer",
                "Paper Fig.2 and Appendix D.2-D.3; author models/kde_particle_filter.py:388,675,812,972; "
                "particle_transformer.py:112; full_sequence_trainer.py:124; weight_model_trainer.py:23; "
                "trainer.py:327,509. Local diagnose_kdm_source_stages_tf.py:55-106 checks source gradients, updates and moments.", stages)

    def sampling():
        root = BASE / "kdm-sampling-20261006-02"
        m, s, author = rd(root / "manifest.json"), rd(root / "summary.json"), rd(root / "author.json")
        ck("native sampler assertions", s["status"] == "PASS" and passing(s, flag="pass_check") and len(s["checks"]) == 28)
        ck("sampler result binding", sha(root / "summary.json") == m["output_sha256"]["summary.json"] and
           sha(root / "author.json") == m["output_sha256"]["author.json"])
        ck("sampler source pin", author["source_commit"] == PIN)
        audit.author_hashes(author["source_hashes"], "author/sampler")
        for rel, expected in m["code_sha256"].items():
            snapshot = root / "source_snapshot" / Path(rel).name
            ck("sampler saved source/" + rel, snapshot.is_file() and sha(snapshot) == expected)
        old = (root / "source_snapshot/kdm_sampling_tf.py").read_text()
        current = (ROOT / "bayesfilter/replication/kdm_sampling_tf.py").read_text()
        normalized = current.replace(
            "def _circular(self, p, bw, uniform, input_valid, concentration=None):",
            "def _circular(self, p, bw, uniform, input_valid):").replace(
            "        if concentration is None:\n            concentration = 1/(bw[:,-1]+tf.constant(1e-8,self.dtype))\n"
            "        kappa=tf.cast(concentration,tf.float64)[None,:,None]",
            "        kappa=tf.cast(1/(bw[:,-1]+tf.constant(1e-8,self.dtype)),tf.float64)[None,:,None]")
        ck("native sampler default arithmetic unchanged",
           ast.dump(ast.parse(normalized)) == ast.dump(ast.parse(old)))
        circular_calls = [c for c in ast.walk(node(ROOT / CORE[1], "SourceKDMResampler", "_assemble"))
                          if isinstance(c, ast.Call) and ast.unparse(c.func) == "self._circular"]
        ck("sampler default caller does not override concentration",
           len(circular_calls) == 1 and len(circular_calls[0].args) == 4 and not circular_calls[0].keywords)
    audit.group("Native categorical, normal and circular mixture draws",
                "Paper Eq.13-15; author kernel_density_estimator.py and von_mises_full_dist.py; "
                "local SourceKDMResampler._sample -> _assemble -> _circular. Moment/frequency screens remain diagnostic.", sampling)

    def clouds_and_boundaries():
        root = BASE / "pretraining-cloud-law-20261013/attempt01"
        m, d = rd(root / "manifest.json"), rd(root / "diagnostic/result.json")
        audit.file_hashes(m["source_hashes"], "cloud current sources")
        ck("cloud assertions", d["status"] == "PASS" and passing(d) and len(d["checks"]) == 138 and not d["flags"])
        ck("cloud moment coverage", d["moments"] == 26 and len(d["between"]) == 26 and d["particles_per_backend"] == 1000000)
        ck("cloud single traces", len(d["trace_counts"]) == 2 and all(t == 1 for t in d["trace_counts"].values()))
        root = BASE / "pretraining-boundaries-20261013/attempt01"
        m, d, controller = rd(root / "manifest.json"), rd(root / "diagnostic/result.json"), rd(root / "controller/result.json")
        audit.file_hashes(m["source_hashes"], "boundary current sources", root / "sources", True)
        ck("boundary assertions", d["status"] == "PASS" and passing(d) and len(d["checks"]) == 271 and len(d["records"]) == 24)
        ck("boundary single traces", len(d["trace_counts"]) == 8 and all(t == 1 for t in d["trace_counts"].values()))
        audit.file_hashes(controller["hashes"], "controller sources")
        ck("controller replay assertions", controller["status"] == "PASS" and passing(controller) and len(controller["checks"]) == 863)
    audit.group("Native composed pretraining clouds, partial batches, late optimizer state and source controller",
                "Paper Appendix D.2-D.3; author DistributionCloudGenerator/WeightModelTrainer and trainer.py/trainer_utils.py; "
                "local SourceWeightPretraining.native_update and SourceEpochController", clouds_and_boundaries)

    def wiring():
        neural, stage = ROOT / CORE[3], ROOT / CORE[4]
        for endpoint, impl in [("self.native_evaluate", "self._native_evaluate"), ("self.update", "self._update")]:
            ck("graph binding/" + endpoint, binding(node(neural, "BearingsNeuralReplay", "_bind_graphs"), endpoint, impl))
        edges = [
            (neural, "BearingsNeuralReplay", "_native_evaluate", ["self._native_sequence"]),
            (neural, "BearingsNeuralReplay", "_native_sequence", ["self._sequence"]),
            (neural, "BearingsNeuralReplay", "_sequence", ["self.sampler._sample", "self.transition", "self.kde._posterior"]),
            (neural, "BearingsNeuralReplay", "transition", ["self.kde._resampling", "self.mlp", "self.bandwidth"]),
            (stage, "SourceInitialCloud", "_sample", ["self.drawer.draw", "self._replay"]),
            (stage, "IndependentNormalVonMises", "draw", ["self.sampler._circular"]),
            (stage, "SourceWeightPretraining", "_native_loss", ["self.center_drawer.draw", "self.cloud_drawer.draw", "self._loss"]),
            (stage, "SourceWeightPretraining", "_loss", ["m.kde._posterior", "m.mlp", "self._select"]),
            (stage, "SourceFullTraining", "_loss", ["self.initial._sample", "self.model._native_sequence"]),
            (stage, "SourceFullTraining", "_update", ["self.model._apply_update"]),
            (stage, "SourceWeightPretraining", "_native_update", ["self.model._apply_update"]),
            (ROOT / "docs/benchmarks/diagnose_kdm_native_evaluation_tf.py", None, "main", ["initial.sample", "model.native_evaluate"]),
            (ROOT / "docs/benchmarks/diagnose_kdm_source_epoch_tf.py", None, "run_epochs",
             ["BearingsNeuralReplay", "SourceFullTraining", "SourceWeightPretraining", "SourceEpochController", "transfer_stage_parameters", "run"])]
        for path, owner, method, targets in edges:
            found = calls(node(path, owner, method))
            ck("calls/" + (owner or path.stem) + "." + method, set(targets) <= found, dict(required=targets))
        epoch = node(ROOT / "docs/benchmarks/diagnose_kdm_source_epoch_tf.py", None, "run_epochs")
        refs = references(epoch)
        ck("epoch selects all four compiled endpoints",
           {"view.native_update", "view.native_evaluate", "view.update", "view.evaluate"} <= refs)
        ck("epoch actually passes selected function to runner",
           any(isinstance(n, ast.Call) and ast.unparse(n.func) == "run" and len(n.args) > 1 and
               ast.unparse(n.args[1]) == "function" for n in ast.walk(epoch)))
        for cls in ("SourceFullTraining", "SourceWeightPretraining"):
            init = node(stage, cls, "__init__")
            endpoint = "self.update" if cls == "SourceFullTraining" else "self.native_update"
            implementation = "self._update" if cls == "SourceFullTraining" else "self._native_update"
            ck("stage compiled binding/" + cls, binding(init, endpoint, implementation))
    audit.group("Actual consumer call chains", "Executable AST wiring plus numerical stage/endpoint evidence; existence of unused methods is insufficient", wiring)

    def endpoints():
        roots = [
            (BASE / "recent-endpoint-20261013/attempt01", ["unit01", "unit02"], [769, 616], [{"mdpf", "am_dpf"}]*2),
            (BASE / "unit02-initialization-20261013/attempt02", ["endpoint"], [472], [{"mdpf"}])]
        for root, subs, totals, variants in roots:
            m = rd(root / "manifest.json")
            audit.file_hashes(m["source_hashes"], root.parent.name + "/sources", root / "sources", True)
            for sub, count, expected_variants in zip(subs, totals, variants):
                result_path = root / sub / "native-result.json"
                d, metrics = rd(result_path), rd(root / sub / "metric-audit.json")
                ck(sub + "/native assertions", d["status"] == "PASS" and passing(d) and len(d["checks"]) == count)
                ck(sub + "/consumer scope", {r["variant"] for r in d["records"]} == expected_variants and
                   len({r["seed"] for r in d["records"]}) == 2 and
                   (d["rows"], d["horizon"], d["particles"], d["batch"]) == (256, 150, 25, 64))
                ck(sub + "/GPU XLA memory", d["jit_compile"] is True and d["tf32"] is False and
                   d["memory_policy"]["all_physical_devices_memory_growth"] is True)
                ck(sub + "/independent metric replay", metrics["status"] == "PASS" and
                   passing(metrics, key="records") and not metrics["failures"] and
                   metrics["native_result_sha256"] == sha(result_path) and metrics["source_pin"] == PIN and
                   len(metrics["records"]) == 2*len(d["metric_replays"]))
                audit.author_hashes(metrics["source_hashes"], sub + "/metric author")
                for replay in d["metric_replays"]:
                    ck(sub + "/replay hash/" + replay["path"],
                       sha(root / sub / replay["path"]) == replay["sha256"])
        terminal = rd(BASE / "unit02-initialization-20261013/attempt02/terminal-review.json")
        ck("intervention baseline unchanged", terminal["exact_baseline_replay"] is True and
           terminal["training_repeated"] is False and terminal["numerical_runtime_changed"] is False)
    audit.group("MDPF/A-MDPF native filters and same-output author metrics",
                "Paper Section5 / Appendix D; author full_sequence_evaluation_base.py and loss_functions.py; "
                "native evaluator -> SourceInitialCloud.sample -> BearingsNeuralReplay.native_evaluate -> NormalVonMisesKDM._posterior. "
                "Nine recent replay clouds/18 metric assertions plus three intervention clouds/six metric assertions.", endpoints)

    def prior_flags():
        d = rd(BASE / "gradient-followup-20261013/pretraining01/diagnostic/result.json")
        ck("pretraining selected-case exact replays", d["status"] == "COMPLETED" and d["exact_reproduction_checks"] == 340 and len(d["selected_replays"]) == 2)
        for row in d["selected_replays"]:
            ck("restored state/" + str(row["index"]),
               bool(row["same_source_state"]) and all(value["max_scaled"] <= 1 and not value.get("flagged", []) for value in row["same_source_state"].values()))
        d = rd(BASE / "gradient-followup-20261013/angle02/diagnostic/result.json")
        m = rd(BASE / "gradient-followup-20261013/angle02/manifest.json")
        audit.file_hashes(m["source_hashes"], "angle intervention sources")
        ck("angle intervention completed", d["status"] == "COMPLETED" and len(d["summaries"]) == 2)
        for name, row in d["summaries"].items():
            control = row["reverse_source_intervention"]
            match = row["interventions"]["all_angle_pairs"]
            ck("angle controls/" + name, len(row["controls"]) == 5 and all(value is True for value in row["controls"].values()) and
               len(control["source_controls"]) == 4 and all(value is True for value in control["source_controls"].values()) and control["zero_control"] is True)
            ck("angle matched-input flag clears/" + name,
               match["localization_flag_removed"] is True and match["ancestor_differences"] == 0 and match["gradients"]["max_scaled"] < 1)
        d = rd(BASE / "gradient-followup-20261013/terminal-audit.json")
        ck("gradient terminal audit", d["status"] == "PASS" and passing(d, flag="pass") and len(d["checks"]) == 115)
        d = rd(BASE / "causal-localization-20261012/saved-case/attempt02/diagnostic/result.json")
        ck("resampling boundary replay", d["status"] == "COMPLETED" and d["passed"] == d["checks"] == 2358 and
           d["capture_change"]["exact_numerical_outputs"] is True)
    audit.group("Previously flagged cases retained and localized",
                "Explanatory case evidence only: restored author state; exact-input angular adjoints; source-only perturbation and same-ancestor replay. No universal perturbation tolerance inferred.", prior_flags)

    tree = git(AUTHOR, "ls-tree", "-r", "--name-only", PIN).splitlines()
    appendix = dict(status="NOT_ESTABLISHED", toy_modules=[p for p in tree if "toy_problem" in p],
                    experiment_families=sorted({p.split("/")[2] for p in tree if p.startswith("src/experiments/")}),
                    interpretation="Generic toy modules exist; mapping to the exact Appendix B temporal-prediction experiment and settings is not established.")
    failures = [c for c in audit.checks if not c["passed"]]
    verdict = "ACCEPTED_BOUNDED_RELEASED_SOURCE_REPLICATION" if accepted(audit) else "BLOCKED_NAMED_EVIDENCE_GAPS"
    wall, cpu = time.monotonic()-started, time.process_time()-cpu_start
    report = dict(schema="kdm_source_coverage_v3", decision=verdict, source_pin=PIN, rows=audit.rows,
                  checks=audit.checks, failed_checks=failures, reviewed_harness_changes=audit.drift,
                  appendix_b=appendix, self_tests=self_test(),
                  scope="Released 25-particle bearings MDPF/A-MDPF implementation, on the tested source operations and consumers.",
                  limitations=["No population noninferiority or independent-training equality claim.",
                               "Moment and categorical-frequency screens are diagnostics, not exhaustive distribution proofs.",
                               "Source-intrinsic numerical sensitivities remain; arbitrary new discrepancies require investigation.",
                               "No whole-paper reproduction, original iAPF, production/HMC readiness or superiority claim."])
    (out / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# KDM released-source closure", "", verdict, "",
             "| Operation group | Verdict |", "|---|---|"] + [
                 "| " + r["operation"] + " | " + r["status"] + " |" for r in audit.rows]
    lines += ["", "Exact Appendix B experiment: NOT_ESTABLISHED. Generic toy modules are present.",
              "", "This decision concerns source-operation fidelity. Independent-training scores remain descriptive.",
              "", "Failed checks: " + str(len(failures)) + ". See coverage.json for every checked dependency and limitation."]
    (out / "coverage.md").write_text("\n".join(lines) + "\n")
    manifest = dict(schema="kdm_closure_manifest_v3", command=[sys.executable, "-I", "-B", *sys.argv],
                    cwd=str(ROOT), git_commit=git(ROOT, "rev-parse", "HEAD"),
                    branch=git(ROOT, "branch", "--show-current"),
                    environment=dict(python=sys.version, executable=sys.executable, CUDA_VISIBLE_DEVICES="-1",
                                     framework_imported=False, cpu_only=True),
                    data_version="Reused hash-bound source fixtures and saved datasets; no new data.",
                    seeds="N/A: deterministic stdlib audit; original numerical seeds remain in linked manifests.",
                    wall_seconds=wall, process_cpu_seconds=cpu, gpu_seconds=0,
                    plan_file=PLAN, result_file=RESULT, evidence_sha256=audit.evidence,
                    audit_source_sha256=sha(Path(__file__)), current_runtime_sha256={p: sha(ROOT/p) for p in CORE},
                    output_sha256={f: sha(out/f) for f in ("coverage.json", "coverage.md")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = dict(decision=verdict, checks=len(audit.checks), failed=[c["name"] for c in failures],
                   operation_groups=len(audit.rows), wall_seconds=wall, process_cpu_seconds=cpu)
    (out / "execution.log").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return 0 if not failures else 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if not args.output:
        parser.error("--output is required for a coverage audit")
    return run(args)

if __name__ == "__main__":
    sys.exit(main())
