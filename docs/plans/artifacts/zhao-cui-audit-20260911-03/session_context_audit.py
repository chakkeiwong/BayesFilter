"""Read-only saved-session analysis; emits sizes and flags, never raw messages."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3


SESSION = Path(
    "/home/chakwong/.codex/sessions/2026/09/10/"
    "rollout-2026-09-10T23-22-12-01a08be9-85ba-7f21-8e53-a2dbf566ab7d.jsonl"
)
LOG_IDS = (456905801, 456915089, 456919752, 456916455, 456917562,
           456918726, 456919916, 456920065, 456923266, 456923537,
           456923607, 456924112, 456924274)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calls, outputs, counts, usage = {}, [], Counter(), []
    phase = "recovery"
    for line_number, line in enumerate(SESSION.open(), 1):
        record = json.loads(line)
        payload = record.get("payload", {})
        kind = payload.get("type")
        counts[f"{record['type']}:{kind}"] += 1
        timestamp = record.get("timestamp", "")
        # The saved scientific continuation starts after 00:38 HKT.
        if timestamp >= "2026-09-10T16:38:00":
            phase = "algorithm_audit"
        if record["type"] == "response_item" and kind in (
            "function_call", "custom_tool_call"
        ):
            content = payload.get("input", payload.get("arguments", ""))
            if not isinstance(content, str):
                content = json.dumps(content)
            calls[payload["call_id"]] = {
                "line": line_number,
                "tool": payload.get("name"),
                "input_characters": len(content),
                "requested_output_budgets": re.findall(
                    r'max_output_tokens["\s]*:\s*(\d+)', content
                ),
                "exec_command_count": content.count("tools.exec_command("),
                "mentions_saved_session": ".codex/sessions" in content,
            }
        if record["type"] == "response_item" and kind in (
            "function_call_output", "custom_tool_call_output"
        ):
            content = payload.get("output", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            outputs.append({
                "line": line_number, "timestamp": timestamp, "phase": phase,
                "call_id": payload.get("call_id"),
                "characters": len(content),
                "truncated": "truncated output" in content.lower()
                or "tokens truncated" in content.lower(),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            })
        if record["type"] == "event_msg" and kind == "token_count":
            info = payload.get("info") or {}
            usage.append({"line": line_number, "timestamp": timestamp,
                          "last_token_usage": info.get("last_token_usage"),
                          "model_context_window": info.get("model_context_window")})

    logs = sqlite3.connect("file:/home/chakwong/.codex/logs_2.sqlite?mode=ro", uri=True)
    selected_logs = []
    for log_id in LOG_IDS:
        row = logs.execute(
            "SELECT ts, feedback_log_body FROM logs WHERE id=?", (log_id,)
        ).fetchone()
        if row is None:
            selected_logs.append({"id": log_id, "missing": True})
            continue
        timestamp, body = row
        selected_logs.append({
            "id": log_id,
            "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "context_fields": re.findall(
                r'(?:auto_compact_scope_tokens|auto_compact_scope_limit|'
                r'full_context_window|full_context_window_limit_reached|'
                r'token_limit_reached)=[^,\s]+', body
            ),
            "upstream_failure": "Upstream request failed" in body,
            "pre_sampling_compaction": "run_pre_sampling_compact" in body,
        })
    logs.close()
    output_ids = {output["call_id"] for output in outputs}
    largest = sorted(outputs, key=lambda entry: entry["characters"], reverse=True)[:8]
    result = {
        "session": str(SESSION), "record_count": sum(counts.values()),
        "scope": "Saved payload character counts, not token counts or billed usage",
        "source_sha256": hashlib.sha256(SESSION.read_bytes()).hexdigest(),
        "calls": len(calls), "outputs": len(outputs),
        "unmatched_calls": sorted(set(calls) - output_ids),
        "unmatched_outputs": sorted(output_ids - set(calls)),
        "successful_compaction_records": sum(
            count for kind, count in counts.items() if kind.startswith("compacted:")
        ),
        "output_characters": sum(output["characters"] for output in outputs),
        "phase_output_characters": {
            part: sum(output["characters"] for output in outputs if output["phase"] == part)
            for part in ("recovery", "algorithm_audit")
        },
        "truncated_outputs": sum(output["truncated"] for output in outputs),
        "largest_outputs": [{**output, "call": calls.get(output["call_id"])}
                            for output in largest],
        "repeated_identical_output_groups": sum(
            count > 1 for count in Counter(output["sha256"] for output in outputs).values()
        ),
        "last_reported_usage": usage[-1] if usage else None,
        "selected_runtime_logs": selected_logs,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as output:
        output.write(serialized)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("largest_outputs", "selected_runtime_logs")}, indent=2))
    print("Largest output summaries:")
    for output in largest[:4]:
        print(json.dumps({"line": output["line"], "characters": output["characters"],
                          "phase": output["phase"], "call": calls.get(output["call_id"])}))


if __name__ == "__main__":
    main()
