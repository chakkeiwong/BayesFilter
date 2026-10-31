"""Resume independent resource-stopped pilots after a bounded diagnosis settles."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    prior = json.loads((args.diagnosis / "manifest.json").read_text())
    command = prior["command"]
    diagnosis_service = command[command.index("--service") + 1]
    while time.time() < config["campaign_started_epoch"] + 42 * 3600:
        state = subprocess.run(["systemctl", "--user", "show", diagnosis_service,
            "--property=ActiveState", "--value"], check=True, capture_output=True, text=True, timeout=10)
        if state.stdout.strip() in {"inactive", "failed"}:
            break
        time.sleep(30)
    else:
        raise RuntimeError("original new-work deadline reached while waiting for diagnosis")
    result = json.loads((args.diagnosis / "result.json").read_text())
    if result["exit_code"] != 0:
        raise RuntimeError("diagnostic coordinator failed; inspect before continuing")
    # A classified target-specific diagnostic failure does not invalidate the
    # six independent resource-stopped fits. The next launcher rechecks source,
    # preflight, settled ledger, common pool and original wall deadlines.
    script = Path(__file__).with_name("run_hmc_ssm_pilot_repair.py")
    launch = [sys.executable, str(script), str(args.config.resolve()), str(args.output.resolve()),
        "--mode", "recover", "--service", args.service, "--gpu", args.gpu]
    raise SystemExit(subprocess.run(launch, check=False).returncode)


if __name__ == "__main__":
    main()
