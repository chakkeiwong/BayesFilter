# Codex session stall investigation

Active question: why session `01a08f5c-e140-7ac1-93af-3e8ab3ac4417`
stopped on the September 12 literature request, and whether context use
contributed. This is a read-only incident investigation, not continuation of
the scientific work. The source session and application databases stay intact.

Checked: the final turn completed its skill read; the app measured 124,873
tokens against a 120,000 automatic-compaction threshold and a 258,400 full
window. Remote compaction failed through two retries, ending with an upstream
stream error. Its last server-reported request total was 111,929 tokens; these
are different counters, not conflicting measurements. Earlier connection
failures also exist and must be separated from this final failure.

Skeptical audit: do not reuse the pasted older 245,203-token/504 diagnosis as
evidence for this event; do not equate characters, cumulative usage, request
usage, replacement history, and guardian history. Tool sizes can establish
avoidable context pressure but cannot prove the server's reason for failure.

Budget: local metadata analysis only; no model invocation, GPU job, environment
change, session resume, or research experiment. Complete: measured 24 tool
outputs / 605,965 saved-text characters since the last successful compaction;
confirmed oversized explicit output requests and repeated policy text. Eight
earlier connection warnings belong to a separate attempt. See `diagnosis.md`,
`session-metrics.json`, and `selected-app-events.json` for evidence and limits.
Next: deliver this incident result; use `research-restart-brief.md` for a fresh
research conversation if the user resumes the literature question. No active
scientific execution is running from this investigation.
