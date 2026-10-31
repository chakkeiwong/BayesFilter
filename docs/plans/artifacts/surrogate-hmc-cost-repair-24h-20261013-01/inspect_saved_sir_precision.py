"""Diagnostic-only analysis of saved precision results; no numerical runtime import."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
paths = [ROOT / "final-precision-01" / ("sir_d18-" + dtype) / "sir_d18/result.json"
         for dtype in ("fp32", "fp64")]
records = [json.loads(path.read_text()) for path in paths]
for key in ("controls", "beta", "data_sha256", "particles", "horizon",
            "parameter_positions", "source_sha256", "source_dependency_sha256"):
    if records[0][key] != records[1][key]:
        raise ValueError("mismatched comparison field: " + key)
fields = records[0]["trace_fields"]
rows = []
for row in range(2):
    candidates = [record["candidate"] for record in records]
    increments = [candidate["score_trace"][row] for candidate in candidates]
    cumulatives = [[[math.fsum(step[p] for step in series[:t]) for p in range(3)]
                    for t in range(1, 51)] for series in increments]
    discrepancy = [max(abs(a-b)/max(1., abs(b))
                       for a, b in zip(cumulatives[0][t], cumulatives[1][t]))
                   for t in range(50)]
    first = next((t+1 for t, gap in enumerate(discrepancy) if gap > .01), None)
    selected_times = sorted(set([1, 5, 10, 20, 30, 40, 50] + ([first] if first else [])))
    temporal = []
    for t in selected_times:
        temporal.append(dict(observation=t, cumulative_scaled_score_gap=discrepancy[t-1],
            incremental_max_abs_score_gap=max(abs(a-b) for a,b in zip(increments[0][t-1], increments[1][t-1])),
            cumulative_scores=[series[t-1] for series in cumulatives],
            diagnostics=[{field: candidate["trace"][row][t-1][fields.index(field)]
                          for field in ("ess", "reset_covariance_relative_error", "sinkhorn_row_error", "reset_margin")}
                         for candidate in candidates]))
    rows.append(dict(parameter_position=records[0]["parameter_positions"][row],
        valid=[candidate["valid"][row] for candidate in candidates],
        final_values=[candidate["value"][row] for candidate in candidates],
        final_scores=[candidate["score"][row] for candidate in candidates],
        first_cumulative_one_percent_crossing=first,
        maximum_cumulative_scaled_score_gap=max(discrepancy),
        reconstruction_max_abs_gap=[max(abs(a-b) for a,b in zip(series[-1], candidate["score"][row]))
                                    for series,candidate in zip(cumulatives,candidates)],
        final_particle_tangent_max_abs=[record["final_particle_tangent_max_abs"][row] for record in records],
        temporal=temporal))
result = dict(status="SAVED_TRACE_LOCALIZATION_COMPLETE", role="explanatory_only",
    utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, str(Path(__file__).resolve())],
    environment="standard library only; no framework import or GPU execution",
    inputs=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
    dtype_order=["fp32", "fp64"], rows=rows,
    nonclaim="Temporal localization does not identify the responsible primitive or establish model-oracle accuracy.")
(ROOT / "sir-fp32-trace-diagnostic-01.json").write_text(json.dumps(result, indent=2)+"\n")
for index,row in enumerate(rows):
    print(json.dumps(dict(row=index, first_crossing=row["first_cumulative_one_percent_crossing"],
        maximum_cumulative_scaled_score_gap=row["maximum_cumulative_scaled_score_gap"],
        reconstruction_max_abs_gap=row["reconstruction_max_abs_gap"],
        final_scores=row["final_scores"])))
