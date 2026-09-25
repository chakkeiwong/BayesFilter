"""Inspect an exact saved Codex session without reproducing its conversations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3


CONTEXT_FIELDS = re.compile(
    r"\b(auto_compact_scope_tokens|auto_compact_scope_limit|"
    r"full_context_window|full_context_window_limit|full_context_window_limit_reached|"
    r"token_limit_reached|total_usage_tokens|estimated_token_count|"
    r"model_context_window|model_auto_compact_token_limit)="
    r"([^,\s]+)"
)


def content_string(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(item.get("text", "") for item in value if isinstance(item, dict))
    return ""


def inspect_session(path):
    counts, calls, outputs, usage, messages, turns = Counter(), {}, [], [], [], []
    completions = []
    digest = hashlib.sha256()
    request = 0
    metadata = {}
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            digest.update(line)
            record = json.loads(line)
            payload = record.get("payload", {})
            record_type, kind = record["type"], payload.get("type")
            counts[f"{record_type}:{kind}"] += 1
            stamp = {"line": line_number, "timestamp_utc": record.get("timestamp")}
            if record_type == "session_meta" and not metadata:
                metadata = {key: payload.get(key) for key in (
                    "id", "timestamp", "cwd", "cli_version", "source",
                    "model_provider", "forked_from_id",
                )}
            if record_type == "turn_context":
                turns.append({**stamp, **{key: payload.get(key) for key in (
                    "turn_id", "model", "effort", "truncation_policy",
                )}})
            if record_type == "event_msg" and kind == "user_message":
                request += 1
            if record_type == "event_msg" and kind == "task_complete":
                completions.append({
                    **stamp, "request": request, "error": payload.get("error"),
                    "duration_ms": payload.get("duration_ms"),
                })
            if record_type == "response_item" and kind == "message":
                content = content_string(payload.get("content", ""))
                messages.append({
                    **stamp, "role": payload.get("role"), "characters": len(content),
                    "instruction_block": "# AGENTS.md instructions" in content,
                    "global_policy_headings": content.count("# Global Scientific Coding Agent Policy"),
                })
            if record_type == "response_item" and kind in ("function_call", "custom_tool_call"):
                content = content_string(payload.get("input", payload.get("arguments", "")))
                pragma = re.search(r"^\s*//\s*@exec:\s*(\{[^\n]+\})", content)
                outer = json.loads(pragma.group(1)) if pragma else None
                budgets = [int(x) for x in re.findall(
                    r"max_output_tokens[\"\s]*:\s*(\d+)", content
                )]
                calls[payload["call_id"]] = {
                    **stamp, "tool": payload.get("name"), "request": request,
                    "input_characters": len(content), "outer_exec_options": outer,
                    "requested_output_budgets": budgets,
                    "exec_command_count": content.count("tools.exec_command("),
                    "mentions_saved_sessions": ".codex/sessions" in content,
                    "mentions_tool_registry": "ALL_TOOLS" in content,
                    "mentions_mathdev": "mathdev" in content.lower(),
                    "mathdev_calls": re.findall(r"tools\.mcp__mathdevmcp__(\w+)\(", content),
                    "mentions_global_instructions": any(x in content for x in (
                        "AGENTS.md", "CLAUDE.md", "global-scientific-coding-agent-policy.md",
                    )),
                    "mentions_json_stringify": "JSON.stringify" in content,
                }
            if record_type == "response_item" and kind in (
                "function_call_output", "custom_tool_call_output",
            ):
                value = payload.get("output", "")
                content = content_string(value)
                decoded_text = text_content(value)
                outputs.append({
                    **stamp, "request": request, "call_id": payload.get("call_id"),
                    "characters": len(content),
                    "decoded_text_characters": len(decoded_text),
                    "truncated": any(x in content.lower() for x in (
                        "truncated output", "tokens truncated", "text items ...]",
                    )),
                    "global_policy_headings": content.count("Global Scientific Coding Agent Policy"),
                    "saved_response_markers": content.count("response_item"),
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                })
            if record_type == "event_msg" and kind == "token_count" and payload.get("info"):
                info = payload["info"]
                usage.append({**stamp, "request": request,
                              "last_token_usage": info.get("last_token_usage"),
                              "model_context_window": info.get("model_context_window")})
                cumulative_usage = info.get("total_token_usage")
    for output in outputs:
        output["call"] = calls.get(output["call_id"])
    output_ids = {output["call_id"] for output in outputs}
    return {
        "session": str(path), "sha256": digest.hexdigest(), "metadata": metadata,
        "scope": "Saved character sizes and reported counters; no character-to-token conversion",
        "record_counts": dict(counts), "calls": len(calls), "outputs": len(outputs),
        "unmatched_calls": sorted(set(calls) - output_ids),
        "successful_compaction_records": sum(v for k, v in counts.items() if k.startswith("compacted:")),
        "output_characters": sum(o["characters"] for o in outputs),
        "decoded_output_text_characters": sum(o["decoded_text_characters"] for o in outputs),
        "truncated_outputs": sum(o["truncated"] for o in outputs),
        "output_characters_by_request": dict(sorted(Counter({
            str(r): sum(o["characters"] for o in outputs if o["request"] == r)
            for r in {o["request"] for o in outputs}
        }).items())),
        "message_sizes": messages, "turns": turns, "usage": usage,
        "completions": completions,
        "last_reported_cumulative_usage": cumulative_usage if usage else None,
        "tool_calls": list(calls.values()), "tool_outputs": outputs,
    }


def inspect_logs(database, thread_id):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    result = []
    rows = connection.execute(
        "SELECT id, ts, level, target, feedback_log_body FROM logs "
        "WHERE thread_id=? ORDER BY ts, ts_nanos, id", (thread_id,),
    )
    for row_id, timestamp, level, target, body in rows:
        body = body or ""
        lower = body.lower()
        fields = dict(CONTEXT_FIELDS.findall(body))
        markers = [marker for marker in (
            "upstream request failed", "stream disconnected", "run_pre_sampling_compact",
            "remote compact", "compaction", "request failed", "connection closed",
            "context length", "status code: 400", "status code: 413",
        ) if marker in lower]
        statuses = re.findall(r"(?:status=|unexpected status )(\d{3})", body)
        if not fields and not markers and not statuses:
            continue
        result.append({
            "id": row_id,
            "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "level": level, "target": target, "fields": fields, "markers": markers,
            "http_statuses": statuses,
            "api_paths": sorted(set(re.findall(r"api.path=\"([^\"]+)\"", body))),
            "during_automatic_compaction": "run_auto_compact" in body,
            "retry": re.findall(r"\b(?:retries|max_retries)=(\d+)", body),
            "request_ids": re.findall(r"(?:request[_ ]id[:= ]+|x-request-id[\" :]+)([a-zA-Z0-9-]{8,})", body),
            "hosts": sorted(set(re.findall(r"https?://([a-zA-Z0-9.-]+)", body))),
        })
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=Path("/home/chakwong/.codex/logs_2.sqlite"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inspect_session(args.session)
    result["selected_runtime_logs"] = inspect_logs(args.database, result["metadata"]["id"])
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps({key: result[key] for key in (
        "metadata", "calls", "outputs", "unmatched_calls", "output_characters",
        "truncated_outputs", "successful_compaction_records", "output_characters_by_request",
    )}, indent=2))
    print(f"Sanitized evidence: {args.output}")


if __name__ == "__main__":
    main()
