# Endpoint and locator execution checkpoint

This checkpoint repairs the target-failure numerical boundary, adds reusable
staged L-BFGS numerical stages, and repairs the reusable batched locator's GPU
accounting declarations. Public staged/batched locator integration is still
gated; the campaign is not complete and main is unmerged.

## Evidence through03327

| Change | Evidence | Limit |
| --- | --- | --- |
| Batched locator accounting |03297 preserves the unmodified original GPU cross-device int32-resource failure. Seven accounting resources and matching integer producers now use int64.03298--03306 pass nine renewed CPU checks against the unmodified d6a568384 reference. |GPU numerical qualification remains pending. The GPU reference adaptation is explicit and integer-only; it is not an unmodified GPU baseline. Public wrapper unchanged. |
| Target-failure boundary |03307/03309 pass57 CPU checks: complete pinned3582b4ac records across scalar/vector/matrix/empty positions, every declared error, exact programmer/shape/permission errors, callback order, changing exception behavior, XLA composition/derivatives and stable HLO. |The arbitrary Python callback remains an explicit host diagnostic adapter. Compiled consumers need tensor failure statuses; this does not make that callback a compiled target. GPU and costs remain open. |
| Staged locator |03311--03320 pass ten complete original-record CPU cases.03321 passes eight existing consumer/wall-boundary checks;03322 passes two construction-exception checks;03323 passes two graph/XLA label smokes;03326 passes complete cross-owner state restoration. Initial/scale and complete optimizer/accounting state are operands. |Internal candidate only. GPU, public integration, native-error/reuse qualification and matched costs remain open. Native compile/runtime failures propagate without an eager retry. |
| Policy and discovery |All129 checks pass03324 and03327. Guard229 sources/1333 exact exceptions; no new waiver. Inventory03325 discovers3048 working-tree Python files,3047 parsed. |The one parse error is the unchanged historical vendor leading-zero literal. Counts are discovery leads, not repository-wide compliance certification. |

Every staged comparison preserves target order/counts, validator timing and
decisions, complete records, one trace per executed stage, unchanged HLO across
changed starts/scales, and actual callback/graph/owner collection. Cap3 at D3
exhausts only after the checkpoint, proving the continuation carries global
accounting. Synthetic consumer cases preserve the best interior callback and
reject a finite sentinel endpoint. The full TFP L-BFGS namedtuple is passed to
continuation rather than restarting at the checkpoint position.

Run03326 resumes the checkpoint on a second owner whose resources first executed
an unrelated start and scale. Full original results and joined checkpoint/
continuation target calls match, showing that this tested continuation restores
its supplied state rather than inheriting the second owner's stale resources.

Run03310 passed numerical comparisons but failed its callback-lifetime
assertion. That fixture shared one Python callback between the original and
candidate, so it could not attribute retention. Independent callback wrappers
isolate the two arms;03311 passes without a numerical change. Preserve03310 as
a harness failure. Construction-error comparisons retain original exception
type, all fields and exact target counts. The JIT report now comes from the
actual function specification;03323's non-JIT arm is a tiny explicit diagnostic.

The original and candidate float arithmetic, scientific settings and numerical
tolerances are unchanged. The target-output cache holds at most16 static-shape
programs and no callback, position or policy. This is a Python ownership bound,
not a native executable-eviction or arbitrary-lifetime memory claim.

## Review and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep batched accounting repair |Nine full CPU cases pass |GPU evidence absent |Device placement after integer repair |GPU records and enclosing recurrence, then E5 integration |GPU/default/public readiness |
| Keep target numerical segment |Exact public records/errors and CPU XLA checks pass |Arbitrary host callback is not a compiled target |GPU/cost qualification and final caller renewal |Qualify GPU; retain explicit adapter boundary |Entire API compilation or HMC readiness |
| Keep internal staged candidate |23 CPU cases pass, including two configuration smokes and cross-owner continuation |Public/GPU/cost gates open |Native runtime errors and public owner reuse |GPU and public/error/ownership qualification |Public replacement or native memory containment |
| Continue master program |Focused progress preserved |AllF01--F20 terminal dispositions open |Initializers, actual DZ5, clean GPU costs and memory dispositions |Follow E2/E4/E5/E6 repair map |Campaign completion or main merge |

Review checked that no callback was silently deleted or traced as a substitute
for its host semantics, no L-BFGS restart replaces continuation, no global cap
resets between stages, and no comparison tolerance or clipping criterion was
relaxed. The weakest evidence remains default GPU qualification and actual
consumer integration. A passing component cannot close those endpoint gates.

Latest fetched remote main is a5aa1fa94. Its new supplied-funnel-map/HMC work
does not touch these runtime/test files; integration/retests still belong to
the terminal phase. No other campaign or external MacroFinance source was
modified. The pending E2 reporting-count proposal remains uninstalled and
requires its specific agreement. Canonical LEDH rebuilding remains excluded.

All commands use the existing campaign runner and fresh numbered directories
under `artifacts/filter-gradient-repair-20260917`. The audit03325 was
inadvertently launched with the runner's GPU device default: its33.395314s stays
charged to GPU, but AST inventory performs no TensorFlow/GPU computation and
is not GPU evidence. Future inventory commands must pass `--device CPU`.
Final charges and per-unit reservations are recorded in the adjacent ledger
and checkpoint receipt; caps remain32CPU/52GPU process-hours.

Receipt: `artifacts/filter-gradient-repair-20260917/endpoint-locator-checkpoint-03327.json`.
Charges are65883.568327 CPU/63625.068103 GPU seconds. Batched locator uses
21/32 workers and482.041581/3600 seconds; target-failure uses3/8 and23.850441/1200;
staged locator uses17/32 and426.276688/2400. Ruff and whitespace checks pass.
The latest GPU preflight at08:17UTC declined after six samples without launching
a worker; it is a resource-contention result, not an approval rejection.
