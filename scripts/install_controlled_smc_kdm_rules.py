"""Install the exact project-scoped replication diagnostic rule set."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/codex_rules/controlled-smc-kdm.rules"
DEST = Path("/home/chakwong/.codex/rules/bayesfilter-controlled-smc-kdm.rules")
REPORT = ROOT / ("docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gap-closure-20261010/allowlist-install-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_text = SOURCE.read_text()
    if not source_text.strip():
        raise RuntimeError("empty rule source")
    REPORT.mkdir(parents=True, exist_ok=False)
    backup = None
    if DEST.exists():
        backup = DEST.with_name(DEST.name + ".before-20261010-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
        shutil.copy2(DEST, backup)
    candidate = DEST.with_suffix(".install-tmp")
    candidate.write_text(source_text)
    flags = []
    for path in sorted(DEST.parent.glob("*.rules")):
        if path != candidate:
            flags.extend(("--rules", str(path)))
    checks = [
        ["/home/chakwong/anaconda3/envs/tftwogpu/bin/python", "/home/chakwong/BayesFilter/docs/benchmarks/run_replication_gap_localization.py", "--output", "/tmp/localization-policy-check"],
        ["/home/chakwong/anaconda3/envs/tftwogpu/bin/python", "/home/chakwong/BayesFilter/docs/benchmarks/capture_lorenz_fit_trace.py", "--output", "/tmp/controlled-smc-c0-check"],
        ["/home/chakwong/anaconda3/envs/tftwogpu/bin/python", "/home/chakwong/BayesFilter/docs/benchmarks/capture_lorenz_forward_reference.py", "--output", "/tmp/controlled-smc-c0-tape-check", "--reference", "/tmp/controlled-smc-reference-check"],
        ["/home/chakwong/anaconda3/envs/tftwogpu/bin/python", "/home/chakwong/BayesFilter/docs/benchmarks/compare_lorenz_fitting_forward_tf.py", "--root", str(ROOT), "--output", "/tmp/controlled-smc-c1-check", "--reference", "/tmp/controlled-smc-reference-check"],
    ]
    for command in checks:
        result = subprocess.run(["codex", "execpolicy", "check", "--rules", str(candidate), *flags, "--", *command], capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        if payload.get("decision") != "allow":
            raise RuntimeError(f"rule did not allow exact form: {payload}")
    negative = subprocess.run(["codex","execpolicy","check","--rules",str(candidate),"--","/home/chakwong/anaconda3/envs/tftwogpu/bin/python","-c","print(0)"],capture_output=True,text=True,check=True)
    if json.loads(negative.stdout).get("decision") == "allow":
        raise RuntimeError("Project rule unexpectedly allows arbitrary Python")
    candidate.replace(DEST)
    (REPORT / "result.json").write_text(json.dumps({
        "status": "PASS",
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "installed": str(DEST),
        "installed_sha256": sha(DEST),
        "backup": str(backup) if backup else None,
        "checks": checks,
        "scope": "four exact replication diagnostic script prefixes",
        "no_broad_rule": True,
        "manual_rule_reload": "required for a future Codex session; not assumed here",
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "installed": str(DEST), "backup": str(backup) if backup else None}))


if __name__ == "__main__":
    main()
