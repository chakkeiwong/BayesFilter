"""Training-only continuation with recoverable state; no new scientific claims."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import tensorflow as tf

from bayesfilter.score_study.moment_training_campaign import save, serializable
from bayesfilter.score_study.neural_kdm_study import NeuralKDMStudy
from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot


RECOVERY_SCHEMA = "sir_neural_kdm_recovery_v1"
PLAN = "docs/plans/ledh-sir-neural-kdm-continuation-20261021.md"


class RecoverableNeuralKDMStudy(NeuralKDMStudy):
    def __init__(self, arguments):
        self.last_recovery_update = -1
        self.latest_recovery = None
        super().__init__(arguments)

    def restore(self, path):
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("recovery_schema") is not None:
            if checkpoint["recovery_schema"] != RECOVERY_SCHEMA \
                    or checkpoint.get("selectable") is not False \
                    or checkpoint["validation"].get("valid") is not False \
                    or checkpoint.get("next_training_seed") != 263041000 + checkpoint["updates"] \
                    or checkpoint.get("next_particle_seed") != 283010000 + checkpoint["updates"]:
                raise ValueError("Invalid recovery checkpoint metadata")
        elif not checkpoint["validation"]["valid"]:
            raise ValueError("Unvalidated state lacks recovery provenance")
        super().restore(path)

    def save_recovery(self):
        if self.completed_updates == self.last_recovery_update:
            return self.latest_recovery
        if int(self.optimizer.iterations) != self.completed_updates:
            raise ValueError("Cannot preserve inconsistent Adam/data state")
        parameters = [tf.convert_to_tensor(variable) for variable in self.network.parameters]
        optimizer = [tf.convert_to_tensor(variable) for variable in self.optimizer.variables]
        if not all(bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float64))))
                   for value in parameters + optimizer):
            raise ValueError("Cannot preserve nonfinite recovery state")
        path = self.output / f"recovery-{self.completed_updates:04d}.json"
        payload = dict(recovery_schema=RECOVERY_SCHEMA, selectable=False,
            configuration=self.configuration(), updates=self.completed_updates,
            parameters=parameters, optimizer=optimizer,
            validation=dict(valid=False, loss=None, role="recovery_only", datasets=0, replicates=0),
            calibration_seed=263040001, reset_seed=263040002,
            next_training_seed=263041000 + self.completed_updates,
            next_particle_seed=283010000 + self.completed_updates,
            source_hashes=self.source["source_hashes"],
            model_score_accuracy_evaluated=False, holdout_opened=False)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(serializable(payload), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        self.last_recovery_update = self.completed_updates
        self.latest_recovery = str(path)
        return self.latest_recovery

    def check_budget(self):
        if self.completed_updates % 8 == 0:
            self.save_recovery()
        try:
            super().check_budget()
        except TimeoutError:
            self.save_recovery()
            raise


def continuation_projection(update_seconds, validation_seconds, updates, setup_reserve=120.):
    if not update_seconds or updates <= 0 or validation_seconds < 0:
        raise ValueError("Missing or invalid continuation timing evidence")
    recent = sorted(update_seconds[-20:])
    if any(not math.isfinite(value) or value <= 0 for value in recent):
        raise ValueError("Invalid update timing evidence")
    upper = recent[math.ceil(.9 * len(recent)) - 1]
    return 1.25 * (updates * upper + validation_seconds) + setup_reserve


def continuation_rungs(starting_updates, target_updates):
    if target_updates not in (256, 512) or not 128 <= starting_updates < target_updates:
        raise ValueError("Unsupported continuation range")
    return [updates for updates in (192, 256, 384, 512)
            if starting_updates < updates <= target_updates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--target-updates", type=int, choices=(256, 512), default=256)
    parser.add_argument("--budget-seconds", type=float, default=3950.)
    parser.add_argument("--campaign-remaining-seconds", type=float, default=4891.197895057)
    arguments = parser.parse_args()
    if not 0 < arguments.budget_seconds <= 3950. \
            or arguments.budget_seconds + 50 > arguments.campaign_remaining_seconds:
        parser.error("Continuation exceeds the reviewed process or campaign budget")
    checkpoint = json.loads(arguments.resume.read_text())
    expected = dict(components=2, width=16, beta=.9, learning_rate=.001, seed=263044001)
    if checkpoint["configuration"] != expected:
        parser.error("Only the predeclared candidate and continuation range are eligible")
    try:
        rungs = continuation_rungs(checkpoint["updates"], arguments.target_updates)
    except ValueError as error:
        parser.error(str(error))
    for name, value in expected.items():
        setattr(arguments, name, value)
    arguments.updates = rungs[0]
    launched = time.monotonic()
    study = RecoverableNeuralKDMStudy(arguments)
    source_path = Path(__file__).resolve()
    wrapper_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (study.output / "continuation-runner.py").write_bytes(source_path.read_bytes())
    study.manifest.update(plan=PLAN, continuation_runner_sha256=wrapper_hash,
        continuation_resume=str(arguments.resume.resolve()),
        continuation_resume_sha256=hashlib.sha256(arguments.resume.read_bytes()).hexdigest(),
        starting_updates=study.completed_updates, requested_rungs=rungs,
        campaign_remaining_seconds=arguments.campaign_remaining_seconds,
        promotion_allowed=False, holdout_opened=False)
    save(study.output / "manifest.json", study.manifest)
    result = dict(status="running", rungs=[], holdout_opened=False,
        model_score_accuracy_evaluated=False, promotion_allowed=False, plan=PLAN)
    returncode = 0
    try:
        study.save_recovery()
        while True:
            begin = time.monotonic()
            rung = study.fit()
            rung["wall_seconds"] = time.monotonic() - begin
            result["rungs"].append(rung)
            save(study.output / f"rung-{study.completed_updates:04d}.json", rung)
            if rung["status"] != "fit_complete" or not rung["validation"]["valid"]:
                result["status"] = "invalid_training_candidate"
                returncode = 2
                break
            if study.completed_updates == arguments.target_updates:
                result["status"] = "bounded_continuation_complete"
                break
            records = [json.loads(line) for line in (study.output / "training.jsonl").read_text().splitlines()]
            duration = [row["wall_seconds"] for row in records][-64:]
            validation_and_overhead = max(0., rung["wall_seconds"] - sum(duration))
            next_rung = next(updates for updates in rungs if updates > study.completed_updates)
            projected = continuation_projection(duration, validation_and_overhead, next_rung - study.completed_updates)
            remaining = arguments.budget_seconds - (time.monotonic() - study.started)
            result["next_rung_projection"] = dict(updates=next_rung, seconds=projected, remaining_seconds=remaining)
            save(study.output / "result.json", result)
            if projected > remaining:
                result["status"] = "bounded_rung_complete_budget_stop"
                break
            arguments.updates = next_rung
        study.save_recovery()
    except TimeoutError as error:
        result.update(status="budget_stop_recoverable", error=repr(error))
        study.save_recovery()
    except Exception as error:
        result.update(status="failed", error=repr(error))
        returncode = 2
    result.update(updates=study.completed_updates, latest_recovery=study.latest_recovery,
        wall_seconds=time.monotonic() - launched, source_check=verify_source_snapshot(study.repository, study.source),
        wrapper_unchanged=hashlib.sha256(source_path.read_bytes()).hexdigest() == wrapper_hash,
        memory=tf.config.experimental.get_memory_info("GPU:0"))
    result["charged_seconds"] = math.ceil(result["wall_seconds"])
    result["wall_seconds_scope"] = "after_imports; outer accounting includes process startup"
    result["remaining_campaign_seconds"] = arguments.campaign_remaining_seconds - result["charged_seconds"]
    if not result["source_check"]["unchanged"] or not result["wrapper_unchanged"]:
        result["status"] = "source_drift"
        returncode = 2
    save(study.output / "result.json", result)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
