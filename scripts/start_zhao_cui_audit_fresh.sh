#!/usr/bin/env bash
set -euo pipefail

audit_workspace=/home/chakwong/BayesFilter
audit_checkout=/home/chakwong/BayesFilterZhaoCui
audit_checkpoint="$audit_workspace/docs/plans/zhao-cui-integrated-audit-restart-2026-09-11.md"

if [[ ! -d "$audit_checkout" || ! -r "$audit_checkpoint" ]]; then
    printf '%s\n' 'Missing Zhao-Cui checkout or audit restart checkpoint.' >&2
    exit 1
fi

audit_prompt="Continue the integrated Zhao-Cui mathematical audit from $audit_checkpoint. Read that brief first; it replaces the need to load recovery reports or old conversations. Preserve unrelated edits. Analyze code and LaTeX, develop coherent repair proposals, and use MathDevMCP for the mathematical assertions. Start with the source inventory stage; runtime repairs remain proposals during this audit. Every functions.exec call must begin with the exact pragma // @exec: {\\\"max_output_tokens\\\":3500}; keep each command at most 2000 output tokens and keep the sum of command allowances below 3500. Save full logs to disk and return only selected result fields and artifact paths. Use explicit checkout paths. Save the bounded stage's findings and update the brief with the next exact task before ending; stop the conversation at the stage boundary rather than waiting for automatic compaction. Follow the plan's existing diagnostic and proof budgets; do not launch a scientific comparison."

audit_command=(
    codex
    -C "$audit_checkout"
    --add-dir "$audit_workspace"
    -c model_auto_compact_token_limit=120000
    -c tool_output_token_limit=2000
    "$audit_prompt"
)

# No resume/fork arguments: the checkpoint supplies state to a fresh thread.
case "${1-}" in
    --print-command)
        printf '%q ' "${audit_command[@]}"
        printf '\n'
        ;;
    '')
        exec "${audit_command[@]}"
        ;;
    *)
        printf '%s\n' 'Usage: bash scripts/start_zhao_cui_audit_fresh.sh [--print-command]' >&2
        exit 2
        ;;
esac
