"""Read-only, standard-library incident analysis; never exports raw prompts."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
import sqlite3
import tomllib

ROOT = Path(__file__).resolve().parent
SID = "01a08f5c-e140-7ac1-93af-3e8ab3ac4417"
SESSION = Path("/home/chakwong/.codex/sessions/2026/09/11/"
               "rollout-2026-09-11T15-27-04-" + SID + ".jsonl")


def chars(value):
    return len(json.dumps(value, ensure_ascii=False))


def display_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(display_text(v) for v in value)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return ""


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def tool_summary(items):
    return {
        "calls_with_output": len(items),
        "display_text_characters": sum(x["display_text_characters"] for x in items),
        "outputs_over_8000_characters": sum(x["display_text_characters"] > 8000 for x in items),
        "outputs_with_truncation_notice": sum(x["truncation_notice"] for x in items),
        "calls_requesting_over_4000_tokens": sum(x["largest_requested_budget"] > 4000 for x in items),
    }


events = [(i, json.loads(line)) for i, line in enumerate(SESSION.open(), 1)]
calls, outputs, turns, compactions, tokens = {}, [], [], [], []
window, current_turn = 0, None
last_compaction_line = 0
for line, event in events:
    p, kind = event.get("payload", {}), event["type"]
    sub = p.get("type")
    if kind == "session_meta":
        meta = {k: p.get(k) for k in ("id", "timestamp", "cwd", "cli_version", "source", "model_provider")}
    if kind == "event_msg" and sub == "task_started":
        current_turn = {"turn_id": p["turn_id"], "start_line": line,
                        "start": event["timestamp"], "outputs": [], "tokens": []}
        turns.append(current_turn)
    if kind == "event_msg" and sub == "task_complete":
        if current_turn and current_turn["turn_id"] == p["turn_id"]:
            current_turn.update(end_line=line, end=event["timestamp"],
                                duration_ms=p.get("duration_ms"), error=p.get("error"),
                                has_final_message=bool(p.get("last_agent_message")))
    if kind == "event_msg" and sub == "token_count":
        info = p.get("info") or {}
        row = {"line": line, "timestamp": event["timestamp"], **info}
        tokens.append(row)
        if current_turn:
            current_turn["tokens"].append(row)
    if kind == "compacted":
        window += 1
        last_compaction_line = line
        row = {"line": line, "timestamp": event["timestamp"], "window": window}
        for key in ("replacement_history", "guardian_history"):
            history = p.get(key) or []
            sizes = Counter()
            fingerprints = Counter()
            summaries = []
            for item in history:
                txt = display_text(item.get("content", []))
                role = item.get("role", item.get("type", "unknown"))
                sizes[role] += len(txt)
                fingerprints[sha256(txt.encode()).hexdigest()] += bool(txt)
                summaries.append({"role": role, "text_chars": len(txt),
                                  "global_policy_banner_count": txt.count("# Global Scientific Coding Agent Policy"),
                                  "context_discipline_heading_count": txt.count("## Context, Tool Output, And Recovery Discipline")})
            row[key] = {"items": len(history), "serialized_chars": chars(history),
                        "message_text_chars_by_role": dict(sizes),
                        "duplicate_nonempty_message_copies": sum(max(0, n - 1) for n in fingerprints.values()),
                        "item_sizes": summaries if key == "replacement_history" else None}
        compactions.append(row)
    if kind != "response_item":
        continue
    if sub in ("custom_tool_call", "function_call"):
        code = str(p.get("input", p.get("arguments", "")))
        budgets = [int(v) for v in re.findall(r'max_(?:output_)?tokens[\\"\s:]+(\d+)', code)]
        category = "other"
        for label, pattern in (("session_investigation", r'\.codex/(sessions|logs|state)|session.*diagnos|context.*incident'),
                               ("literature_tool", r'mathdev|researchassistant'),
                               ("web", r'web__|search_query'),
                               ("local_read_or_command", r'exec_command'),
                               ("file_edit", r'apply_patch')):
            if re.search(pattern, code, re.I):
                category = label
                break
        calls[p.get("call_id")] = {"call_line": line, "name": p.get("name"),
                                  "category": category, "input_characters": len(code),
                                  "has_exec_pragma": "@exec:" in code,
                                  "largest_requested_budget": max(budgets, default=0)}
    if sub in ("custom_tool_call_output", "function_call_output"):
        txt = display_text(p.get("output", ""))
        row = {**calls.get(p.get("call_id"), {}), "output_line": line,
               "timestamp": event["timestamp"], "window": window,
               "display_text_characters": len(txt),
               "serialized_output_characters": chars(p.get("output")),
               "truncation_notice": bool(re.search(r'tokens truncated|Warning: truncated output|Output truncated', txt)),
               "display_text_sha256": sha256(txt.encode()).hexdigest()}
        outputs.append(row)
        if current_turn:
            current_turn["outputs"].append(row)

turn_rows = []
for t in turns:
    ts = t.pop("tokens")
    os = t.pop("outputs")
    t["tools"] = tool_summary(os)
    t["first_recorded_request_tokens"] = ts[0].get("last_token_usage") if ts else None
    t["last_recorded_request_tokens"] = ts[-1].get("last_token_usage") if ts else None
    turn_rows.append(t)

conn = sqlite3.connect("file:/home/chakwong/.codex/logs_2.sqlite?mode=ro", uri=True, timeout=5)
log_rows = []
for lid, ts, level, target, body in conn.execute(
        "SELECT id,ts,level,target,feedback_log_body FROM logs WHERE thread_id=? ORDER BY ts,id", (SID,)):
    body = body or ""
    excerpt = None
    for marker in ("post sampling token usage", "remote compaction v2 stream failed", "stream connection failed"):
        if marker in body:
            excerpt = body[body.index(marker):]
            break
    if excerpt:
        log_rows.append({"log_id": lid, "timestamp": utc(ts), "level": level,
                         "target": target, "excerpt": excerpt})
final_start = int(datetime.fromisoformat(turn_rows[-1]["start"].replace("Z", "+00:00")).timestamp())
final_end = int(datetime.fromisoformat(turn_rows[-1]["end"].replace("Z", "+00:00")).timestamp())
final_http_504_mentions = conn.execute(
    "SELECT count(*) FROM logs WHERE thread_id=? AND ts BETWEEN ? AND ? "
    "AND feedback_log_body LIKE '%504%'", (SID, final_start, final_end)).fetchone()[0]
conn.close()
config = tomllib.loads(Path("/home/chakwong/.codex/config.toml").read_text())
summary = {
    "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "source": {"path": str(SESSION), "bytes": SESSION.stat().st_size,
               "sha256": sha256(SESSION.read_bytes()).hexdigest(), "lines": len(events)},
    "metadata": meta,
    "current_config_not_historical_proof": {k: config[k] for k in (
        "model", "model_auto_compact_token_limit", "tool_output_token_limit", "model_context_window") if k in config},
    "whole_session_tools": tool_summary(outputs),
    "since_last_compaction_tools": tool_summary([o for o in outputs if o["output_line"] > last_compaction_line]),
    "largest_tool_outputs": sorted(outputs, key=lambda x: x["display_text_characters"], reverse=True)[:12],
    "turns": turn_rows, "successful_compactions": compactions,
    "last_recorded_tokens": tokens[-1],
    "final_turn_thread_log_504_mentions": final_http_504_mentions,
    "all_tool_outputs": outputs,
    "interpretation_limits": [
        "Serialized and display-text character counts are not token counts.",
        "Display text can itself contain JSON-escaped nested outputs; no recursive double decoding is used.",
        "Historical aggregate tool volume is not currently active context.",
        "Guardian history is recorded separately and is not assumed to be ordinary active prompt input.",
        "Cumulative usage includes repeated context processing; cached input is part of input tokens.",
        "No model reasoning text, raw instructions, credentials, or full session content is exported.",
    ],
}
(ROOT / "session-metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
(ROOT / "selected-app-events.json").write_text(json.dumps(log_rows, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("source", "whole_session_tools", "since_last_compaction_tools")}, indent=2))
print("Saved session-metrics.json and selected-app-events.json")
