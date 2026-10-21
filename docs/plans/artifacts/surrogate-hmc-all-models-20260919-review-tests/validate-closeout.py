"""CPU-only source/document/CLI validation; no numerical or inference claims."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
documents = [ROOT / "docs/plans" / name for name in (
    "surrogate-hmc-all-models-master-program-2026-09-18.md",
    "surrogate-hmc-all-models-master-review-2026-09-18.md",
    "surrogate-hmc-all-models-claude-handoff-2026-09-18.md",
    "surrogate-hmc-all-models-execution-2026-09-19.md",
    "surrogate-hmc-all-models-execution-result-2026-09-19.md",
)]
missing = []
link_count = 0
for path in documents:
    for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text()):
        if target.startswith(("http:", "https:", "#")):
            continue
        link_count += 1
        target = target.split("#", 1)[0]
        if not (path.parent / target).exists():
            missing.append([str(path.relative_to(ROOT)), target])
assert not missing, missing
sources = [ROOT / path for path in (
    "bayesfilter/inference/surrogate_hmc_program.py",
    "bayesfilter/testing/surrogate_hmc_lg1_tf.py",
    "scripts/run_surrogate_hmc_program.py",
    "scripts/surrogate_hmc_lg1_probe.py",
    "tests/test_surrogate_hmc_program.py",
)]
for path in sources:
    compile(path.read_text(), str(path), "exec")
run = OUT.parent / "surrogate-hmc-all-models-20260919-review-02"
manifest = json.loads((run / "run-manifest.json").read_text())
hash_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
archive_match = {relative: hash_file(OUT / "source" / Path(relative).name) == expected
                 for relative, expected in manifest["source_sha256"].items()}
assert all(archive_match.values()), archive_match
changed_after_run = [relative for relative, expected in manifest["source_sha256"].items()
                     if hash_file(ROOT / relative) != expected]
assert changed_after_run == ["scripts/surrogate_hmc_lg1_probe.py"], changed_after_run

with tempfile.TemporaryDirectory(prefix="surrogate-hmc-closeout-") as directory:
    preview = Path(directory) / "preview"
    command = [sys.executable, "scripts/run_surrogate_hmc_program.py", "--output", str(preview), "--dry-run"]
    dry = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    (OUT / "final-dry-run.json").write_text(dry.stdout)
    payload = json.loads(dry.stdout)
    assert not preview.exists() and not payload["inference_ready"]
    assert len(payload["prior_runs"]) == 3
    rejection = subprocess.run([sys.executable, "scripts/surrogate_hmc_lg1_probe.py",
                                "--output", str(preview), "--particle-count", "16"],
                               cwd=ROOT, text=True, capture_output=True)
    assert rejection.returncode == 2 and "invalid choice" in rejection.stderr
    assert not preview.exists()
    (OUT / "invalid-particle-count.log").write_text(rejection.stderr)
decision = json.loads((run / "decision.json").read_text())
assert decision["status"] == "INCOMPLETE_IMPLEMENTATION"
assert not decision["all_models_validated"] and not decision["default_readiness"]
assert set(decision["comparisons"].values()) == {"NOT_RUN"}
assert decision["heuristic_dominance"] == "NOT_RUN"
report = dict(status="PASS", classification="CPU-only structural checks; no inference evidence",
              gpu_intentionally_hidden=True, local_links_checked=link_count,
              sources_compiled=len(sources), missing_links=missing,
              executed_source_archive_matches=archive_match,
              changed_after_run=changed_after_run,
              post_run_change="Probe CLI now rejects N other than the declared N1024; executed numerical path unchanged",
              dry_run_no_files_or_inference=True, cumulative_prior_runs=3,
              wrong_particle_count_rejected_before_framework_import=True,
              source_sha256={str(path.relative_to(ROOT)):hash_file(path) for path in sources},
              document_sha256={str(path.relative_to(ROOT)):hash_file(path) for path in documents})
(OUT / "final-validation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps({key:report[key] for key in ("status","local_links_checked","sources_compiled","cumulative_prior_runs")}))
