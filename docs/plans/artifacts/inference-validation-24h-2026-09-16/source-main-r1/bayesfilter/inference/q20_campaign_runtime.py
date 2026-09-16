"""Standard-library process supervision and accounting for local q20 work.

The external timeout survives loss of the coordinator. One ordinary advisory
process lock prevents duplicate coordinators. No launch approval tokens.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bayesfilter.inference.q20_production_config import digest


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_snapshot(repo):
    repo = Path(repo)
    paths = sorted((repo / "bayesfilter").rglob("*.py"))
    paths += sorted((repo / "bayesfilter/ops").glob("*.so"))
    paths.append(repo / "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py")
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _process_start(pid):
    try:
        # The process name may contain spaces or parentheses; fields follow
        # the final ')'. Linux starttime is field 22 (index 19 in this suffix).
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class CampaignBudgetError(RuntimeError):
    pass


class Campaign:
    def __init__(self, root, *, repo, config, allowance=None):
        self.root, self.repo = Path(root).resolve(), Path(repo).resolve()
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "campaign.json"
        self._lock = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["config_hash"] != digest(config) or self.state["sources"] != source_snapshot(self.repo):
                raise ValueError("campaign source/protocol changed; do not resume old numerical state")
        else:
            if allowance is None:
                raise ValueError("a new campaign requires the existing allowance record")
            campaign = float(allowance["campaign_remaining_seconds"])
            diagnostic = float(allowance["diagnostic_remaining_seconds"])
            holds = allowance.get("unsettled_holds", [])
            held = sum(float(row["seconds"]) for row in holds)
            if not (0 <= held < diagnostic <= campaign) or not all(math.isfinite(v) for v in (campaign, diagnostic, held)):
                raise ValueError("invalid allowance/hold arithmetic")
            self.state = {"schema": "bayesfilter.q20.campaign.v1", "created_at": utc_now(),
                "config_hash": digest(config), "config": config, "sources": source_snapshot(self.repo),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip(),
                "allowance": allowance, "campaign_limit": campaign-held, "diagnostic_limit": diagnostic-held,
                "spent_seconds": 0., "diagnostic_spent_seconds": 0., "attempts": [],
                "stages": {}, "status": "initialized", "production_qualified": False}
            # Initialization is completed under the same lock as all writes.

    @contextmanager
    def locked(self):
        with (self.root / "coordinator.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock = lock
            try:
                if self.path.exists():
                    current = json.loads(self.path.read_text())
                    if current["config_hash"] != self.state["config_hash"] or current["sources"] != self.state["sources"]:
                        raise ValueError("campaign changed during coordinator acquisition")
                    self.state = current
                self.save()
                self.recover()
                yield self
            finally:
                self._lock = None
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def save(self):
        if self._lock is None:
            raise RuntimeError("campaign mutations require the coordinator lock")
        atomic_json(self.path, self.state)

    def remaining(self, diagnostic=False):
        value = self.state["campaign_limit"] - self.state["spent_seconds"]
        if diagnostic:
            value = min(value, self.state["diagnostic_limit"] - self.state["diagnostic_spent_seconds"])
        return max(0., value)

    def recover(self):
        for attempt in self.state["attempts"]:
            if attempt["status"] != "running":
                continue
            start = _process_start(attempt.get("pid") or -1)
            if start is not None and start == attempt.get("process_start"):
                raise RuntimeError("previous owned timeout/worker still running; wait for its recorded deadline")
            if attempt.get("pid") is None and time.time() < attempt["started_epoch"] + attempt["cap_seconds"]:
                # The coordinator may have died between Popen and saving PID.
                # Never launch a second worker during the unresolved deadline.
                raise RuntimeError("worker creation was interrupted; wait for the recorded deadline")
            if attempt.get("pid") is not None and start is None:
                self._cleanup_group(attempt["pid"])
            receipt = Path(attempt["directory"]) / "supervisor.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                self._settle(attempt, saved["elapsed_seconds"], saved["status"], saved["returncode"])
            else:
                # External timeout bounded the orphan attempt. With no parent
                # receipt reserve the full cap, explicitly as an upper charge.
                self._settle(attempt, attempt["cap_seconds"], "interrupted_upper_charge", None)
                attempt["accounting_basis"] = "full_external_cap_after_coordinator_loss"
            self.save()

    def _settle(self, attempt, elapsed, status, returncode):
        if attempt["status"] != "running":
            return
        elapsed = max(0., float(elapsed))
        self.state["spent_seconds"] += elapsed
        if attempt["diagnostic"]:
            self.state["diagnostic_spent_seconds"] += elapsed
        attempt.update(status=status, elapsed_seconds=elapsed, returncode=returncode, ended_at=utc_now())

    @staticmethod
    def _cleanup_group(pid):
        # GNU timeout and its command share this owned session. A command may
        # exit while leaving descendants; remove those even after exit code 0.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def stage_remaining(self, stage):
        spent = sum(a.get("elapsed_seconds", 0.) for a in self.state["attempts"] if a["stage"] == stage)
        return max(0., self.config["budget"]["arm_cap_seconds"] - spent)

    def execute(self, stage, command, *, cap_seconds, diagnostic, environment=None, request=None, request_hash=None):
        """Launch one owned process tree; charge wall time on every exit path."""
        if self._lock is None:
            raise RuntimeError("coordinator lock required")
        if source_snapshot(self.repo) != self.state["sources"]:
            raise ValueError("execution sources drifted")
        cap = float(cap_seconds)
        if not math.isfinite(cap) or cap <= 0 or cap > self.remaining(diagnostic):
            raise CampaignBudgetError("requested worker cap exceeds the remaining allowance")
        cap = min(cap, self.stage_remaining(stage))
        if cap <= self.config["execution"]["termination_grace_seconds"]:
            raise CampaignBudgetError("cumulative stage cap exhausted; a retry cannot renew it")
        grace = min(self.config["execution"]["termination_grace_seconds"], cap/2)
        number = len(self.state["attempts"])
        folder = self.root / "attempts" / f"{number:05d}-{stage}"
        folder.mkdir(parents=True, exist_ok=False)
        if request is not None:
            atomic_json(folder / "request.json", request)
        argv = [str(x).replace("{attempt}", str(folder)) for x in command]
        argv = ["timeout", "--signal=TERM", f"--kill-after={grace}s", f"{cap-grace}s", *argv]
        env = {**os.environ, **(environment or {})}
        attempt = {"stage": stage, "status": "running", "directory": str(folder),
            "command": argv, "diagnostic": bool(diagnostic), "cap_seconds": cap,
            "started_at": utc_now(), "started_epoch": time.time(), "pid": None,
            "process_start": None, "request_hash": request_hash,
            "accounting_basis": "supervisor_measured_worker_wall"}
        self.state["attempts"].append(attempt)
        self.state["status"] = "running:" + stage
        self.save()
        started, process = time.monotonic(), None
        try:
            with (folder / "console.log").open("wb") as output:
                process = subprocess.Popen(argv, cwd=self.repo, env=env, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True)
                attempt.update(pid=process.pid, process_start=_process_start(process.pid))
                self.save()
                while process.poll() is None:
                    time.sleep(min(self.config["execution"]["poll_seconds"], .5))
                code = process.returncode
            status = "completed" if code == 0 else "timed_out" if code in (124,137,-9,-15) else "failed"
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if process is not None:
                self._cleanup_group(process.pid)
            status, code = "interrupted", None if process is None else process.returncode
            elapsed = time.monotonic()-started
            atomic_json(folder / "supervisor.json", {"elapsed_seconds": elapsed, "status": status, "returncode": code})
            self._settle(attempt, elapsed, status, code)
            self.save()
            raise
        self._cleanup_group(process.pid)
        elapsed = time.monotonic()-started
        atomic_json(folder / "supervisor.json", {"elapsed_seconds": elapsed, "status": status, "returncode": code})
        self._settle(attempt, elapsed, status, code)
        self.save()
        return attempt

    def numerical_stage(self, name, request, *, cap_seconds, diagnostic, gpu=None):
        existing = self.state["stages"].get(name)
        if existing is not None:
            if existing["request_hash"] != digest(request):
                raise ValueError("completed stage request changed")
            result_path = Path(existing["result_path"])
            data = result_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != existing["result_sha256"]:
                raise ValueError("completed stage result changed")
            for path, checksum in existing["artifact_hashes"].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
                    raise ValueError("completed stage artifact changed: " + path)
            return {**json.loads(data), "supervisor_seconds":existing["supervisor_seconds"]}
        env = {"TF_FORCE_GPU_ALLOW_GROWTH": "true", "PYTHONUNBUFFERED": "1",
               "BAYESFILTER_PRELOAD_CUSTOM_OP":"0"}
        if self.config["cpu_reference"]:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        elif gpu is None:
            raise ValueError("serious numerical stage requires a trusted GPU selection")
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        original_hash = digest(request)
        previous = [a for a in self.state["attempts"] if a["stage"] == name]
        for item in previous:
            if item.get("request_hash") != original_hash:
                raise ValueError("interrupted stage request changed")
        resume = {}
        if previous:
            data_root = Path(previous[-1]["directory"]) / "worker/data"
            kind = request["stage"]
            if kind == "train":
                checkpoints = sorted(data_root.glob("cohort-*.json"))
                if checkpoints:
                    resume["resume_checkpoint"] = str(checkpoints[-1])
            elif kind == "tune" and (data_root / "tuning/tuning_checkpoint.json").is_file():
                resume["resume_checkpoint"] = str(data_root / "tuning/tuning_checkpoint.json")
            elif kind in {"reference", "sample", "ensemble", "replica_exchange"} and (data_root / "chunks").is_dir():
                resume["resume_chunks"] = str(data_root / "chunks")
        worker_config = json.loads(json.dumps(self.config))
        if request.get("role") and worker_config["role"] != "smoke":
            worker_config["role"] = request["role"]
        request = {**request, **resume, "config": worker_config, "plan_file": "docs/plans/bayesfilter-ssl-lstm-q20-executable-master-repair-plan-2026-09-16.md"}
        if request["stage"] == "train":
            request["cooperative_seconds"] = max(0., cap_seconds - 2*self.config["execution"]["termination_grace_seconds"])
        command = [sys.executable, "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py",
                   "worker", "--request", "{attempt}/request.json", "--output-dir", "{attempt}/worker"]
        attempt = self.execute(name, command, cap_seconds=cap_seconds, diagnostic=diagnostic,
                               environment=env, request=request, request_hash=original_hash)
        attempt["request_hash"] = original_hash
        self.save()
        path = Path(attempt["directory"]) / "worker/worker-result.json"
        if attempt["status"] != "completed" or not path.exists():
            self.state["status"] = "stage_"+attempt["status"]+":"+name
            self.save()
            return {"status": self.state["status"], "completed": False, "attempt": attempt}
        data = path.read_bytes()
        result = json.loads(data)
        partial_training = request["stage"] == "train" and not result.get("result", {}).get("cohort_complete", False)
        partial_tuning = request["stage"] in {"tune","reverify"} and result.get("result",{}).get("status") == "paused_infrastructure"
        if result.get("completed") and not partial_training and not partial_tuning:
            self.state["stages"][name] = {"request_hash": original_hash,
                "result_path": str(path), "result_sha256": hashlib.sha256(data).hexdigest(),
                "supervisor_seconds":attempt["elapsed_seconds"],
                "artifact_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(attempt["directory"]) / "worker").rglob("*") if p.is_file()}}
            self.save()
        if partial_training:
            result.update(completed=False, status="partial_training_checkpointed")
        if partial_tuning:
            result.update(completed=False, status="partial_tuning_checkpointed")
        return {**result, "supervisor_seconds":attempt["elapsed_seconds"]}
