"""Independent diagnostic of a complete normal-conjugate fit's posterior mean.

NumPy/SciPy are diagnostic reference authorities here. This module is never
imported by tuning or posterior runtime. The normal approximation defines an
alarm; its achieved false-alarm and detection rates require independent fits.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from scipy import stats

from ..designs import digest, seed_for
from ..storage import read_json, read_tensor, write_json
from .normal_endpoint import normal_reference
from .statistics import binomial_interval

DETECTOR = "normal_conjugate_per_fit_reference_mean_lugsail.v1"
OUTPUT_RULE = "first_verified_id_qualified_retained_model_mean.v1"


def independent_lugsail(values, *, batch_size=None, min_batches=20,
                        lugsail_r=3, lugsail_c=.5):
    """Separate complete-batch reference preserving independent chain axes."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 1 or x.shape[0] < 4 or x.shape[1] < 2:
        raise ValueError("reference mean requires [draw>=4, chain>=2, 1]")
    if not np.all(np.isfinite(x)):
        raise ValueError("reference mean requires finite samples")
    n, chains, _ = x.shape
    b = math.isqrt(n) if batch_size is None else batch_size
    if (type(b) is not int or b < 1 or type(min_batches) is not int or min_batches < 2
            or type(lugsail_r) is not int or lugsail_r < 1
            or type(lugsail_c) not in (int, float)
            or not math.isfinite(lugsail_c) or not 0 <= lugsail_c < 1):
        raise ValueError("invalid independent lugsail settings")
    small = b // lugsail_r
    report = {"estimate": float(x.mean()), "mcse": None, "available": False,
              "batch_size": b, "small_batch_size": small, "batch_count": n // b,
              "unused_terminal_draws": n % b, "per_chain_long_run_variance": None,
              "reason": "insufficient_complete_batches"}
    if small < 1 or n // b < min_batches or n // small < min_batches:
        return report

    def lrv(width):
        count = n // width
        means = x[:count * width, :, 0].reshape(count, width, chains).mean(axis=1)
        return width * np.var(means, axis=0, ddof=1)

    variance = (lrv(b) - lugsail_c * lrv(small)) / (1 - lugsail_c)
    report["per_chain_long_run_variance"] = variance.tolist()
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        report["reason"] = "nonpositive_or_nonfinite_chain_variance"
        return report
    report.update(available=True, reason="available",
                  mcse=math.sqrt(float(variance.sum()) / (n * chains**2)))
    return report


def alarm_from_pipeline(fit_id, pipeline, *, alpha, mcse_sd_max, tau, sigma,
                        estimator_options=None):
    """Read only the frozen selected member; never choose by posterior results."""
    if (type(alpha) not in (int, float) or not 0 < alpha < 1
            or type(mcse_sd_max) not in (int, float)
            or not math.isfinite(mcse_sd_max) or mcse_sd_max <= 0):
        raise ValueError("explicit finite alarm size and precision target required")
    mean, sd = normal_reference(pipeline["data"], tau=tau, sigma=sigma)
    record = {"fit_id": fit_id, "detector": DETECTOR, "output_rule": OUTPUT_RULE,
              "alpha": alpha, "mcse_sd_max": mcse_sd_max,
              "reference_mean": mean, "reference_sd": sd, "status": "unavailable",
              "alarm": None, "z": None, "stream": None, "candidate_id": None,
              "data": pipeline["data"]}
    selection = pipeline["selection"]
    verified = sorted(pipeline["verified_candidate_ids"])
    expected = verified[:1]
    if (selection["rule"] != "first_verified" or selection["scope"] != "selected"
            or list(selection["candidate_ids"]) != expected):
        raise ValueError("reference mean requires the predetermined first-verified member")
    if not expected:
        record["reason"] = "no_verified_member"
        return record
    members = [m for m in pipeline["members"] if m["candidate_id"] == expected[0]]
    if len(members) != 1:
        raise ValueError("missing or duplicated selected member")
    member = members[0]
    posterior = member.get("posterior", {})
    stream = posterior.get("config", {}).get("retained_seed")
    record.update(candidate_id=expected[0], stream=list(stream) if stream is not None else None)
    qualified = (pipeline["completion"] == "complete" and member["status"] == "assessed"
        and posterior.get("passed") is True and posterior.get("warmup_cap_hit") is False
        and posterior.get("retained_cap_hit") is False and not posterior.get("hard_vetoes", [])
        and member.get("warmup_exclusion_matches") is True)
    if not qualified:
        record["reason"] = "missing_capped_or_unqualified_posterior"
        return record
    variance = independent_lugsail(read_tensor(member["draws_path"]), **(estimator_options or {}))
    record["independent_precision"] = variance
    if not variance["available"]:
        record["reason"] = variance["reason"]
        return record
    ratio = variance["mcse"] / sd
    record["mcse_sd_ratio"] = ratio
    if ratio > mcse_sd_max:
        record["reason"] = "independent_precision_target_not_met"
        return record
    z = (variance["estimate"] - mean) / variance["mcse"]
    cutoff = float(stats.norm.ppf(1 - alpha / 2))
    record.update(status="qualified", reason="qualified", z=z, cutoff=cutoff,
                  alarm=abs(z) > cutoff)
    return record


def summarize_alarms(records, planned_fit_ids, *, control, alpha, mcse_sd_max):
    """Exact rate bounds with conservative, full-inventory missingness."""
    planned = tuple(planned_fit_ids)
    if (not planned or any(not isinstance(i, str) or not i for i in planned)
            or len(set(planned)) != len(planned)):
        raise ValueError("distinct nonempty planned fit IDs required")
    if control not in {"baseline", "noop", "location_shift"}:
        raise ValueError("unsupported reference-mean control")
    indexed, streams = {}, set()
    for row in records:
        if (row["fit_id"] not in planned or row["fit_id"] in indexed
                or row["detector"] != DETECTOR or row["output_rule"] != OUTPUT_RULE
                or row["alpha"] != alpha or row["mcse_sd_max"] != mcse_sd_max):
            raise ValueError("unknown/duplicate fit or changed detector policy")
        if row["status"] not in {"qualified", "unavailable"}:
            raise ValueError("unknown reference-mean status")
        if row["status"] == "qualified":
            if (type(row["z"]) not in (int, float) or not math.isfinite(row["z"])
                    or type(row["alarm"]) is not bool
                    or row["alarm"] != (abs(row["z"]) > float(stats.norm.ppf(1 - alpha / 2)))):
                raise ValueError("inconsistent qualified alarm")
            stream = tuple(row["stream"] or ())
            if len(stream) != 2 or any(type(v) is not int for v in stream) or stream in streams:
                raise ValueError("distinct retained fit streams required")
            streams.add(stream)
        elif row["alarm"] is not None:
            raise ValueError("unavailable evidence cannot be an alarm")
        indexed[row["fit_id"]] = row
    qualified = [r for r in records if r["status"] == "qualified"]
    missing = len(planned) - len(qualified)
    alarms = sum(r["alarm"] for r in qualified)
    null = control in {"baseline", "noop"}
    numerator = alarms + missing if null else alarms
    interval = binomial_interval(numerator, len(planned))
    passed = interval[1] <= .10 if null else interval[0] >= .80
    return {"detector": DETECTOR, "output_rule": OUTPUT_RULE, "control": control,
            "alpha": alpha, "mcse_sd_max": mcse_sd_max, "planned": len(planned),
            "recorded": len(records), "qualified": len(qualified), "unavailable": missing,
            "alarms": alarms, "conservative_numerator": numerator, "interval": interval,
            "screen": "null_upper_at_most_0.10" if null else "defect_lower_at_least_0.80",
            "rate_screen_passed": passed, "availability_interval": binomial_interval(len(qualified), len(planned)),
            "missing_policy": "possible_null_alarm_or_defect_nondetection",
            "finding": ("calibration_incomplete" if missing else
                        "discrepancy_detected" if alarms else "no_discrepancy_detected"),
            "independent_unit": "fresh data and complete preparation/tuning/posterior fit",
            "independence_limit": "stream uniqueness is necessary, not proof of statistical independence",
            "power_established": False, "ranking_supported": False,
            "interpretation": "pointwise 95% exact rate screen for this frozen detector only",
            "records": records, "unrecorded_fit_ids": [i for i in planned if i not in indexed]}


def unavailable_record(design, replication, reason):
    return {"fit_id": str(replication), "detector": DETECTOR, "output_rule": OUTPUT_RULE,
            "alpha": design.alpha,
            "mcse_sd_max": design.options["reference_mean_alarm"]["mcse_sd_max"],
            "status": "unavailable", "alarm": None, "z": None, "stream": None,
            "reason": reason}


def run_replication(design, root, replication, deadline=None):
    """One independently generated dataset and one complete public fit."""
    from ..execution import source_state
    from ..procedures import execute_pipeline
    from ..references import analytic

    params = design.scenario.parameters
    data_seed = seed_for(design.seed, design.design_id, replication, "data")
    truth, data = analytic.simulate("normal_conjugate", data_seed, params)
    path = Path(root) / f"replication-{replication:04d}"
    # Only observations reach the fitter; truth and the independent
    # conditional reference never select members or modify tuning.
    pipeline = execute_pipeline(design, path, data=data, dataset_id=replication,
                                fit_id=0, deadline=deadline, reuse_leapfrog_graphs=True)
    row = alarm_from_pipeline(str(replication), pipeline, alpha=design.alpha,
        mcse_sd_max=design.options["reference_mean_alarm"]["mcse_sd_max"],
        tau=params["tau"], sigma=params["sigma"],
        estimator_options=design.options.get("posterior_precision_settings"))
    row.update(data_seed=list(data_seed), truth_assessor_only=truth,
               data_identity=digest(data), source_identity=source_state()["identity"])
    write_json(path / "independent_assessment.json", row)
    return row


def summarize_replications(design, records):
    """Process failure cannot promote an otherwise qualified saved assessment."""
    from ..execution import source_state

    rows = []
    failures = 0
    for record in records:
        row = dict(record)
        if "execution_failure" in row:
            failures += 1
            row = {**unavailable_record(design, row["fit_id"], "execution_failed"),
                   "saved_numerical_assessment": record,
                   "execution_failure": record["execution_failure"]}
        elif list(record["data_seed"]) != list(seed_for(
                design.seed, design.design_id, int(record["fit_id"]), "data")):
            raise ValueError("reference-mean generated-data seed changed")
        rows.append(row)
    result = summarize_alarms(rows, [str(i) for i in range(design.replications)],
        control=design.scenario.control, alpha=design.alpha,
        mcse_sd_max=design.options["reference_mean_alarm"]["mcse_sd_max"])
    result.update(design_identity=design.identity, source_identity=source_state()["identity"],
                  phase=design.phase, tested_procedure="complete_ordinary_pipeline",
                  completed_fits=len(records)-failures, execution_failures=failures,
                  all_fits_recorded=len(records) == design.replications,
                  assessment_complete=result["unavailable"] == 0 and not failures)
    return result


def run(design, root, deadline=None):
    from ..execution import source_state

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"design": design.identity, "source": source_state()["identity"],
                "detector": DETECTOR, "output_rule": OUTPUT_RULE}
    identity_path = root / "reference_mean_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("reference-mean source or design identity changed")
    write_json(identity_path, identity)
    rows = []
    for i in range(design.replications):
        if deadline is not None and time.monotonic() >= deadline:
            break
        row = run_replication(design, root, i, deadline)
        write_json(root / f"alarm-{i:04d}.json", row)
        rows.append(row)
    result = summarize_replications(design, rows)
    write_json(root / "reference_mean.json", result)
    return result
