#!/usr/bin/env python3
"""Test the q* reparameterization by exact chain-rule transformation.

This consumes preserved physical-score artifacts. It does not rerun the
finite-particle kernel: at a fixed physical DGP, the value is invariant and
the transformed score is determined exactly by the Jacobian.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHI = (0.72, 0.55, 0.35)
Q_SCALE = 0.35
R_SCALE = 0.45
DATASET_THETA = (*PHI, Q_SCALE, R_SCALE)
LABELS = ("value", "phi1", "phi2", "phi3", "qstar", "r_scale")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def qstar_geometry() -> dict[str, Any]:
    a_terms = (
        1.0 / (1.0 - PHI[0] ** 2),
        1.0 / (1.0 - PHI[1] ** 2),
        1.0 - PHI[2] ** 2,
    )
    a = sum(a_terms) / 3.0
    da_dphi = (
        2.0 * PHI[0] / (3.0 * (1.0 - PHI[0] ** 2) ** 2),
        2.0 * PHI[1] / (3.0 * (1.0 - PHI[1] ** 2) ** 2),
        -2.0 * PHI[2] / 3.0,
    )
    sqrt_a = math.sqrt(a)
    return {
        "a_terms": list(a_terms),
        "a": a,
        "sqrt_a": sqrt_a,
        "qstar": Q_SCALE * sqrt_a,
        "da_dphi": list(da_dphi),
    }


def transform_physical_score(score: list[float]) -> list[float]:
    geometry = qstar_geometry()
    a = geometry["a"]
    da_dphi = geometry["da_dphi"]
    q_score = float(score[3])
    phi_scores = [
        float(score[index]) - q_score * Q_SCALE * da_dphi[index] / (2.0 * a)
        for index in range(3)
    ]
    return [*phi_scores, q_score / geometry["sqrt_a"], float(score[4])]


def hmc_chain_qstar() -> list[float]:
    geometry = qstar_geometry()
    return [
        *(1.0 - phi * phi for phi in PHI),
        geometry["qstar"],
        R_SCALE,
    ]


def _interval(values: list[float], critical: float = 3.036283222821165) -> dict[str, float]:
    if len(values) != 16:
        raise ValueError("expected exactly 16 claim seeds")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "critical_value": critical,
        "lower": mean - critical * se,
        "upper": mean + critical * se,
    }


def _scope(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load(path)
    result = payload["claim"]["result"]
    geometry = qstar_geometry()
    chain = hmc_chain_qstar()
    transformed_truth = transform_physical_score(result["kalman_physical_score"])
    truth_hmc = [value * scale for value, scale in zip(transformed_truth, chain, strict=True)]
    transformed_rows = [
        transform_physical_score(row) for row in result["per_seed_physical_score"]
    ]
    hmc_rows = [
        [value * scale for value, scale in zip(row, chain, strict=True)]
        for row in transformed_rows
    ]
    value_rows = [float(value) for value in result["per_seed_value"]]
    kalman_value = float(result["kalman_value"])
    candidate_columns = [
        value_rows,
        *([row[index] for row in hmc_rows] for index in range(5)),
    ]
    truth_columns = [kalman_value, *truth_hmc]
    relative_rows = [
        [(candidate - truth) / abs(truth) for candidate, truth in zip(row, truth_columns, strict=True)]
        for row in zip(*candidate_columns, strict=True)
    ]
    intervals = {
        label: _interval([row[index] for row in relative_rows])
        for index, label in enumerate(LABELS)
    }
    original_q_hmc = [float(row[3]) * Q_SCALE for row in result["per_seed_physical_score"]]
    qstar_hmc = [row[3] for row in hmc_rows]
    return {
        "label": label,
        "num_particles": result["num_particles"],
        "seeds": result["estimator_seeds"],
        "geometry": geometry,
        "truth": {
            "physical_theta": list(DATASET_THETA),
            "physical_qstar": geometry["qstar"],
            "physical_score_qstar": transformed_truth[3],
            "hmc_score": truth_hmc,
        },
        "candidate": {
            "mean_value": statistics.mean(value_rows),
            "mean_hmc_score": [statistics.mean(column) for column in candidate_columns[1:]],
            "mean_physical_score_qstar": statistics.mean(row[3] for row in transformed_rows),
            "value_difference_to_original": statistics.mean(value_rows) - kalman_value,
        },
        "relative_error_intervals": intervals,
        "invariance_checks": {
            "qstar_hmc_equals_original_log_q_hmc": all(
                math.isclose(left, right, rel_tol=1.0e-14, abs_tol=1.0e-14)
                for left, right in zip(qstar_hmc, original_q_hmc, strict=True)
            ),
            "value_target_unchanged": True,
            "physical_qstar_relative_bias_equals_original_q_relative_bias": (
                math.isclose(
                    intervals["qstar"]["mean"],
                    _interval(
                    [
                        (row[3] * Q_SCALE - result["kalman_physical_score"][3] * Q_SCALE)
                        / abs(result["kalman_physical_score"][3] * Q_SCALE)
                        for row in result["per_seed_physical_score"]
                    ]
                    )["mean"],
                    rel_tol=1.0e-14,
                    abs_tol=1.0e-14,
                )
            ),
        },
        "source_artifact": str(path),
        "source_artifact_sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n5000-claim", type=Path, required=True)
    parser.add_argument("--n10000-claim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "bayesfilter.lgssm_qstar_reparameterization_diagnostic.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_complete",
        "definition": "qstar^2=q_scale^2*(1/(1-phi1^2)+1/(1-phi2^2)+(1-phi3^2))/3",
        "interpretation": {
            "value": "invariant at fixed physical theta; any change is numerical implementation behavior",
            "qstar_log_score": "identical to original log(q_scale) score by chain rule",
            "phi_scores": "change because q_scale changes when qstar is held fixed",
            "qstar_physical_score": "rescaled by 1/sqrt(A), so relative bias is unchanged",
        },
        "scopes": {
            "N5000": _scope(args.n5000_claim.resolve(), label="N5000"),
            "N10000": _scope(args.n10000_claim.resolve(), label="N10000"),
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "geometry": qstar_geometry()}, indent=2))


if __name__ == "__main__":
    main()
