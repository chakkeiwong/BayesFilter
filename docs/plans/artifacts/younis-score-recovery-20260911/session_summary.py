"""Read-only session diagnostic; saves sizes/counters, never raw conversations."""

import collections
import datetime
import json
from pathlib import Path
import sqlite3

THREAD = "01a06730-c524-7a22-911f-2c6ff57b6a2f"
ROLLOUT = Path("/home/chakwong/.codex/sessions/2026/09/03/"
               f"rollout-2026-09-03T20-14-04-{THREAD}.jsonl")
counts = collections.Counter()
outputs = []
calls = {}
compactions = []
tokens = []
last_tasks = []
for line_number, line in enumerate(ROLLOUT.open(), 1):
    record = json.loads(line)
    kind = record.get("type")
    payload = record.get("payload", {})
    subtype = payload.get("type")
    counts[kind] += 1
    position = {"line": line_number, "utc": record.get("timestamp")}
    if kind == "compacted":
        compactions.append(position)
    if kind == "response_item":
        call_id = payload.get("call_id")
        if subtype in ("function_call", "custom_tool_call"):
            calls[call_id] = {**position, "name": payload.get("name")}
        if subtype in ("function_call_output", "custom_tool_call_output"):
            output = str(payload.get("output", ""))
            outputs.append({**position, "call_id": call_id,
                            "characters": len(output),
                            "contains_truncation_marker": "truncat" in output.lower()})
    if kind == "event_msg" and line_number >= 37365:
        if subtype == "token_count":
            info = payload.get("info") or {}
            tokens.append({**position, "last_token_usage": info.get("last_token_usage"),
                           "model_context_window": info.get("model_context_window")})
        if subtype in ("task_started", "task_complete", "turn_aborted"):
            last_tasks.append({**position, "type": subtype})

def output_stats(start):
    selected = [item for item in outputs if item["line"] >= start]
    returned = {item["call_id"] for item in selected}
    return {
        "start_line": start,
        "output_count": len(selected),
        "output_characters": sum(item["characters"] for item in selected),
        "outputs_with_truncation_marker": sum(item["contains_truncation_marker"] for item in selected),
        "largest_outputs": sorted(selected, key=lambda item: item["characters"], reverse=True)[:6],
        "unreturned_calls": [item for key, item in calls.items()
                             if item["line"] >= start and key not in returned],
    }

database = sqlite3.connect("file:/home/chakwong/.codex/logs_2.sqlite?mode=ro", uri=True, timeout=5)
start_time = int(datetime.datetime.fromisoformat("2026-09-10T20:27:00+00:00").timestamp())
end_time = start_time + 33 * 60
runtime = []
for ident, timestamp, target, body in database.execute(
    "SELECT id,ts,target,feedback_log_body FROM logs WHERE thread_id=? "
    "AND ts BETWEEN ? AND ? AND (level IN ('WARN','ERROR') "
    "OR feedback_log_body LIKE '%post sampling token usage%') ORDER BY id",
    (THREAD, start_time, end_time),
):
    markers = ("post sampling token usage", "remote compaction v2 stream failed",
               "stream disconnected - retrying sampling request", "Failed to run pre-sampling compact")
    marker = next((item for item in markers if item in body), None)
    if marker:
        runtime.append({"id": ident, "utc": datetime.datetime.fromtimestamp(
            timestamp, datetime.timezone.utc).isoformat(), "target": target,
            "message": body[body.index(marker):][:1200]})

summary = {
    "thread": THREAD, "rollout": str(ROLLOUT), "rollout_bytes": ROLLOUT.stat().st_size,
    "records": line_number, "record_counts": counts,
    "last_successful_compactions": compactions[-4:],
    "pasted_exchange": output_stats(37365),
    "holistic_question_and_continuation": output_stats(37394),
    "context_counters": tokens, "task_events": last_tasks, "runtime_evidence": runtime,
    "limitations": ["Characters are not tokens.",
                    "File size and lifetime totals are not the active context size.",
                    "Local logs do not identify the server-side cause of stream failures."],
}
destination = Path(__file__).with_name("session-summary.json")
destination.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"saved": str(destination),
                  "pasted_exchange": summary["pasted_exchange"],
                  "last_runtime_counter": [item for item in runtime if
                                           item["message"].startswith("post sampling")][-1:]}))
