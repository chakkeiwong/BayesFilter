# Codex compaction investigation — 2026-09-14

## Active checkpoint

Question: why the observation-aware TT research session repeatedly stalls during
compaction, whether tool use contributes, and why other local threads continue.
Scope is read-only incident investigation; no research continuation, model/API
probes, configuration changes, session/database edits, or process termination.
Checkout verified: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`; preserve
the existing four tracked modifications and unrelated untracked work.

The attachment matches session `01a091d6-769a-7901-90e0-80900daba20e` exactly,
including the 73.7-second representation diagnostic progress message. Source:
`/home/chakwong/.codex/sessions/2026/09/12/rollout-2026-09-12T02-59-06-01a091d6-769a-7901-90e0-80900daba20e.jsonl`.
Its two terminal turns report remote-compaction stream failures at
2026-09-13 20:42:18 and 20:51:25 UTC. The saved research checkpoint is
`docs/plans/observation-aware-tt-active-checkpoint.md` and predates the campaign
launch shown in the conversation; it must not be assumed current.

Skeptical preflight: distinguish serialized file size, decoded text characters,
active-context tokens, and cumulative usage. Inspect the exact stalled thread;
compare only clearly identified local controls. Neither a successful tool call
nor a successful compact event proves the next compaction can succeed. A remote
failure is not proof of a defective GPU or a stopped experiment. Do not print
raw saved sessions, instruction payloads, HTTP bodies, or credentials.

Evidence directory: `docs/plans/artifacts/codex-compaction-20260914-01/`.
Budget: bounded local parsing and indexed read-only log queries; no numerical
compute or paid model requests. Official documentation search and open returned
502; direct retrieval returned 403 both normally and with escalated permissions.
Use checked local runtime evidence and mark unavailable product guidance.

Investigation complete. The final reported input usage jumped from 58,972 to
130,862 tokens after a short sleep; the runtime counter crossed 120,000 and
remote compaction failed through all automatic and manual retries. Broad reads
and duplicated instructions contribute pressure, but two same-client/gateway
neighbors contain more large results and no comparable token jump. The exact
provider-side cause remains unverified. No configuration or scientific code
was changed. Evidence and remedies are in
`docs/plans/codex-compaction-investigation-20260914-result.md`.

The research run separately failed its pair conditional finite/bracket/CDF
check after 151.322 seconds. The partial result's RUNNING label is stale.
Current recovery pointer:
`docs/plans/observation-aware-tt-recovery-checkpoint-20260914.md`.

Next action: report findings to the owner. Any later research continuation
should start from the recovery note, assess the validity failure under the
existing plan, and reconcile its remaining numerical budget.
