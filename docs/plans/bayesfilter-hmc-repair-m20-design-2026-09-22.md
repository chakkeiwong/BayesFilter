# M20: distinguish numerical search failure from posterior precision

This is the next resolved experiment under the refreshed HMC master. Its
ceiling is 18000 CPU-reference and 18000 GPU worker-seconds, including failures,
pilots, tests and repairs. First-stage jobs below reserve 4800 CPU and 7200 GPU
seconds. These ceilings are convenience allocations, not statistical power.
Use fresh directories below `artifacts/hmc-repair-master-2026-09-16/m20-r1`.

## Research intent and comparators

M17's centered-funnel preparation completed. Its CPU search rejected all eight
pairs, and its GPU search rejected all six; nonfinite proposals account for
most terminal failures. The question is whether a finite smaller-epsilon grid
under identical geometry and starts can supply numerically verified candidates,
and whether that finding transfers to a fresh ordinary preparation. A different
parameterization is a later diagnostic if the centered law remains unsuitable.
It does not retroactively repair the centered-coordinate result.

The first comparator uses the original M16 source snapshot that issued M17's
frozen geometry. It loads that checkpoint through the checked loader, verifies
the source and tensor hashes, then **issues new scopes** with the same physical
start bank, geometry and policy, fresh declared seeds and zero transferred
qualification. Run the original native search and an explicit grid with
epsilon multipliers `(1/8, 1/4, 1/2, 1)` at every original L. Multipliers are
geometric exploration hypotheses motivated by proposal overflow, not evidence
that acceptance is monotone. Both arms retain the original finite search domain,
repair factor, evidence rungs and candidate cap; the grid uses the same 300
work-unit cap. Predeclared refinement is two rounds with failed-interval
exploration enabled in the grid arm; record this as a combined exploration
repair, not an isolated effect of shrinking. All old failures stay unchanged.

Next repeat both arms after one fresh preparation on merged current source,
for the same scale-3 funnel. Use the M17 preparation settings: standard preset,
startup cap 20, finite-window metric evidence with 16 probe draws, three bounded
preparation restarts and one search-bound expansion. Prepare once per pair,
then freeze the identical handoff for both searches. Target log density and
score remain the exact centered funnel. Separate failure in preparation,
finite measurement, acceptance compatibility and verification.

The second question concerns members of rotated Gaussian (condition 100, angle
0.6) and noncentered funnel (scale 3). Repeat fresh ordinary tuning on merged
source, keep every verified member and predeclare the first candidate by ID
at each of L `(3,9,18,25)` before examining any posterior. If a declared L has
no verified member, record it as unavailable; do not substitute a favorable
member after inspecting results. Save the selection before posterior execution.
All selected members use independent streams and the same M17 precision and
posterior policy: 0.05 absolute mean/median MCSE, four chains, 2000 minimum
warmup, a 1000 recent window, 500 chunks and 10000 warmup/retained caps. This is
a clustered development experiment, not four independent successful fits.

## Diagnostic roles and completion

| Diagnostic | Role |
| --- | --- |
| Public binding, target/geometry parity, complete telemetry, fresh streams and exact candidate identity | Engineering pass criterion; missing or corrupt evidence stops and repairs the affected experiment |
| Candidate numerical health and unchanged acceptance compatibility screen | Tuning promotion criteria/vetoes; every survivor requires fresh verification |
| Posterior R-hat, ESS, declared precision and target truth | Separate posterior promotion criteria/vetoes; never changes tuning membership |
| Quantity-level MCSE, inferred lag dependence, acceptance, duration and member counts | Explanatory; one paired development fit cannot support a ranking |
| Empty candidate set, warmup cap or precision cap | Repair trigger; continue to the predeclared next diagnostic, not a reason to stop the program |

The engineering repair is demonstrated only if candidate exploration remains
bounded and independently verified with intact identity. Numerical usability
requires a nonempty set on fresh fits; posterior readiness additionally requires
all separate posterior checks. Failure is preserved, not renamed success.
If the grid succeeds, wire a reusable optional exploration setting only if the
public explicit grid is insufficient for consumers, and add health/lineage/
budget/resume tests before fresh confirmation. If it fails, inspect candidate
health and compare a checked noncentered law; do not widen acceptance or remove
nonfinite vetoes. Test Gaussian and beta-binomial integrations for any changed
shared mechanism. The next phase refresh must record which part is closed.

For precision, compare requested units with exact stationary scale. The Gaussian
median iid planning approximation and funnel child variance are derived in
M19's saved-failure audit. They diagnose affordability; they do not authorize
relaxing 0.05 or increasing the repository cap. A member meeting precision may
be usable for its checked quantities, without being statistically superior.

## Commands, defaults and safeguards

The phase worker is `m20-r1/run_geometry_probe.py`; arguments declare `--source`,
`--device`, `--case`, `--output`, `--seconds` and `--seed`. The coordinator
`m20-r1/run_phase.py` launches the fixed inventory, records each exact command,
source snapshot, hardware and memory policy, elapsed time, return status and
result. Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. CPU workers hide
GPUs before import and use one intra/inter-op thread, explicitly as reference
development arms. GPU workers require trusted execution, visible GPU, XLA and
verified memory growth before initialization. No eager/pfor training is involved.
One GPU worker runs at a time; CPU work uses at most two independent workers.

Root seeds `(2026092201,2026092202)` are convenience-chosen disjoint development
identifiers for old-geometry and fresh-preparation arms. Deterministic derived
streams separate target/device/arm/member/stage. Do not substitute seeds after a
failure. Candidate evidence uses the inherited 128 base draws and eight discarded
transitions; its finite start dependence is preserved, not claimed stationary.
All numerical constants above are inherited baselines or explicitly labeled
exploration/cost hypotheses. The controller's compatibility rule is not a strict
stationary acceptance-band estimator. No default numerical policy changes.

Pre-mortem: a smaller step may pass locally but fail in the narrow funnel neck;
retain independent verification and downstream health. A favorable member may
merely be a lucky stream; select before sampling and replicate before promotion.
Median precision can be harder than mean precision on a broad Gaussian; retain
both. Noncentered child means have heavy Monte Carlo tails; finite R-hat cannot
prove their CLT approximation. Saved-source probes cannot certify merged-source
behavior; keep their manifests and conclusions distinct.

Skeptical audit passed with limits: matched arms share geometry but not posterior
data, repaired search uses a documented combination of grid and refinement,
and source scopes are renewed. Failed candidate numerical evidence never
acquires an acceptance direction. Limits and stopping conditions are explicit.
The first-stage inventory is exploratory and answers mechanism/cost questions;
fresh confirmation and any next implementation are resolved from its actual
failure record under the remaining allocation. No broad calibration, ranking,
universal convergence or new default is concluded from this phase.
