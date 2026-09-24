#!/usr/bin/env python3
"""Install the reviewed, fixed q20 command rules and activate its supervisor."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path("/home/ubuntu/python/BayesFilter")
ROOT = REPO / "docs/plans/artifacts/q20-master-operations-2026-09-21"
RULES = Path("/home/ubuntu/.codex/rules/default.rules")
PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
CONTROL = str(REPO / "scripts/q20_campaign_control.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    block = (ROOT / "q20-campaign.rules").read_text()
    original = RULES.read_text()
    if "# BEGIN BayesFilter q20 campaign control" in original and block not in original:
        raise ValueError("a different q20 command-rule block already exists; inspect before editing")
    combined = original if block in original else original.rstrip()+"\n\n"+block
    candidate = Path("/tmp/q20-combined-rules-check-20260921.rules")
    candidate.write_text(combined)
    matches = []
    for action in ("status", "ensure"):
        direct = [PYTHON, CONTROL, action]
        for command in (direct, ["/bin/bash", "-lc", " ".join(direct)],
                        ["/bin/bash", "-c", " ".join(direct)]):
            result = subprocess.run(["codex", "execpolicy", "check", "--rules", str(candidate), "--", *command],
                                    capture_output=True, text=True, check=True)
            value = json.loads(result.stdout)
            if value.get("decision") != "allow":
                raise ValueError("effective rules do not allow a prepared command: " + str(command))
            matches.append({"command": command, "decision": value["decision"]})
    backup = RULES.with_name("default.rules.before-q20-control-20260921")
    if combined != original:
        if not backup.exists():
            with backup.open("x") as stream:
                stream.write(original)
        temporary = RULES.with_name("default.rules.q20-install.tmp")
        temporary.write_text(combined)
        temporary.chmod(RULES.stat().st_mode & 0o777)
        temporary.replace(RULES)
    receipt = {"installed_at_utc": datetime.now(timezone.utc).isoformat(), "rules_file": str(RULES),
        "backup": str(backup), "before_sha256": sha(original.encode()), "after_sha256": sha(combined.encode()),
        "added_rules": 0 if original == combined else 6,
        "scope": "fixed q20 status and ensure, direct argv and exact bash wrappers",
        "unrelated_rules_preserved": True, "effective_local_checks": matches,
        "managed_reviewer_or_sandbox_changed": False}
    (ROOT / "installation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    # Use exactly the approved argv, without shell assignments or substitutions.
    result = subprocess.run([PYTHON, CONTROL, "ensure"], check=True, capture_output=True, text=True)
    receipt["activation"] = json.loads(result.stdout)
    (ROOT / "installation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"rules_installed": receipt["added_rules"],
        "activation": receipt["activation"]["action"],
        "unit": receipt["activation"].get("unit"),
        "rules_file": str(RULES)}, indent=2))


if __name__ == "__main__":
    main()
