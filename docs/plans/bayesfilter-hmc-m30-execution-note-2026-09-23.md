# M30 execution: compiled runner reuse

Status: bounded execution complete and audited.
Plan: `bayesfilter-hmc-post-m29-next-phase-2026-09-23.md`.
Opening balance from M29: 74979.67865046971 CPU and 77294.48905148171 GPU
worker-seconds. Ceilings: 1800 CPU / 4800 GPU seconds, no more than two
numerical workers and one GPU worker at once. Outputs: `artifacts/`
`hmc-repair-master-2026-09-16/m30-r1/`. Prior authorization covers this phase.

## Skeptical audit before implementation

The question is whether compiled-runner reuse can reduce complete-fit cost
while preserving the numerical experiment. Existing caching already shares
epsilon/seed/state inputs at fixed L/count. Dynamic L is a hypothesis; the
expected failure is changed graph arithmetic, seed folding, or target reuse.
Saved M29 chunks include actual prepared geometry and all states and traces,
so exact replay can answer that engineering question without repeating M29.

Two flaws in the inherited plan are corrected before execution. CPU and GPU
arithmetic need not be bitwise equal: the tiny CPU test compares repeated CPU
calls; GPU/XLA replays must match the archived GPU tensors exactly. Also,
`work_seed` binds the entire scope and candidate hash, including absolute
source paths and checksums. A source edit changes fresh-fit seeds. Stripping
source fields in an audit would conceal that mismatch. Cross-source exact
chunk replay remains valid because it supplies the recorded seeds. A complete
paired fit needs a same-source static/dynamic comparison, to be resolved after
the replay diagnostic within the existing allocation. No source attestation
will be bypassed to manufacture a pair.

Exact numerical and complete-trace equality is the promotion criterion for
the engineering mechanism. Missing/corrupt evidence, wrong reconstruction,
changed seed lineage, invalid memory policy or uncontrolled tracing veto the
affected comparison until repaired. A slower valid prototype rejects its cost
hypothesis, not HMC or the research direction. Timing/RSS/allocator readings
are explanatory; R-hat/ESS/MCSE remain posterior-only. No single fit establishes
speed superiority, stopping coverage, global exploration or default readiness.

The audit passes with these corrections. M29's L=3/25 and counts=8/136/256
are measured cache keys, chosen to cover both endpoints and every actual chunk
size. Three identical calls per cell are a convenience repeatability check,
not statistical replication. Starts, epsilon, seeds, data and preparation
layers come from saved evidence. CPU non-XLA execution is a tiny debug
exception; GPU/XLA, float64 and verified growth remain the actual comparator.
The 300-second replay and 1500-second fit caps are inherited development
ceilings, not adequacy or timing claims. Every failed launch remains charged.
Short bookkeeping has a declared 60-second CPU allowance as in M29.

## Research intent and decision rules

| Item | Rule |
| --- | --- |
| Baseline | M29 source-r3, exact saved observation chunks, existing independent scalar-chain runner |
| Mechanism | Share compiled graphs across L only within a frozen target/geometry/shape/trace/backend and fixed count |
| Repair trigger | Reproducible first-call overhead plus exact dynamic-L replay |
| Promotion veto | Changed numerical tensors, health/acceptance traces, streams or pipeline decisions |
| Continuation veto | Invalid comparison/evidence, scope change or exhausted allowance |
| Next evidence | Focused isolation/restart tests and paired complete public-pipeline fits if replay qualifies |
| Not concluded | Scientific calibration, learned-map quality, general posterior validity or stochastic speed ranking |

All source snapshots use committed code plus listed owned overlays. Unrelated
Q20/NeuTra and governance changes remain excluded. Terminal work must include
the independent audit, receipts, ledger, master refresh and executable next
phase; successful launch alone does not complete M30.

## Baseline diagnostic and prototype decision

Gaussian passed all six GPU cells, three identical calls each, comparing every
state and complete trace against M29. First calls took 10.32--14.24 seconds;
warmed calls took 0.037--0.507 seconds. This nominates graph reuse as an
engineering hypothesis; it is not a stochastic performance ranking. The
beta-binomial baseline is executing separately. Proceed with the narrow
independent-chain wrapper extension, preserving the four scalar graphs and
their original folded seeds, then replay before changing the binding cache.

Three diagnostic setup failures remain charged: a wrong resource-helper
import; an unjustified CPU/GPU prepared-signature equality check (the CPU
smoke now records both signatures and only compares CPU repeats); and a
tuple/list difference in the serialized GPU device provenance check. GPU
prepared signatures and complete tensors still require exact equality.

If dynamic replay passes, add an explicit optional execution-config field,
default false, for count-keyed graph reuse within one binding. It is execution
provenance, excluded from the numerical seed policy but included in the full
binding/config payload. Run static and dynamic public pipeline arms from the
same frozen source, passing the option as an execution argument outside the
design identity. Record the option and compare every output after normalizing
only declared cache telemetry, clocks and binding hashes. This resolves the
cross-source seed flaw without bypassing source validation or replaying an old
source attestation. No global numerical default changes in M30.

## Exact GPU replay passed; full-fit allocation refreshed

Both targets passed all six static and six dynamic cells with three exact
repeats per cell. Dynamic L reused three count-specific runner groups instead
of six L/count groups, with one trace per scalar runner. Warmed dynamic calls
were slower than static calls in these diagnostics, so first-call savings alone
cannot support a complete-fit cost claim. Full unprofiled fits are required.

The interim ledger charges 250.59145502909087 GPU seconds, including the
failed preflight comparison. Within the unchanged 4800-second ceiling,
allocate four outer full-fit attempts capped at 1000 seconds each (980-second
inner budget), leaving 549.4085449709091 seconds for local infrastructure
repair. The extra two same-source static fits are necessary because original
M29 source-dependent streams cannot be reused for a new implementation. All
four use unchanged M29 design files and the same new frozen source directory;
the explicit reuse option is outside design identity and is saved in both
execution config and manifest. No monkeypatch or source-check bypass is used.

The comparison must first establish equal source files, complete target and
geometry bindings, design, seeds and all candidate scopes. Permitted differences
are only the explicit reuse option, its cache/call-count telemetry, clocks,
output paths and dependent integrity hashes (each checked independently).
Every numerical observation, receipt, saved tensor, acceptance/momentum/health
trace and posterior decision must match. Default false remains available;
M30 evaluates the optional path rather than changing a numerical default.

## Terminal checkpoint

All four full fits exited normally. Same-source static/dynamic pairs matched
exactly over 315 complete numerical observations/receipts and 178 saved tensor
records. Gaussian retained 24 verified members from 94 candidates; beta-binomial
retained 16 from 100. All four selected posteriors passed their declared checks.
Other members remain explicitly unassessed, never removed by posterior results.
Gaussian outer cost was 627.135/346.588 seconds static/dynamic; beta-binomial
was 575.165/285.510. These are descriptive single-pair timings.

The 120-case focused regression set passed after correcting two new fixture
allocations that illegally set the repair reserve to zero. Additional suites
passed 73 integration cases, 5 route cases, 24 runner-compatibility cases and
15 documentation cases; suites overlap. The final completed-fit relabeling
guard differs from the numerical snapshot only in its already-complete branch;
the checked three-line delta is preserved in `post-fit-source-delta.json`.
Its focused regression and integration checks passed without repeating fresh
fits. The official 574-page book rebuilt with BibTeX, no unresolved citations
or references, and a visual check of PDF page 424 (printed 406). An adjacent
overfull class-name line was repaired without changing its meaning.

The independent terminal audit verifies raw saved tensor fingerprints against
M29 archives, source differences, exact same-source full-fit pairs, full
geometry, candidate scopes, seed lineage, receipts, all posterior decisions,
memory growth, device/XLA provenance and normal exits. No numerical tolerance
was introduced. All 30 outer attempts are terminal. M30 charges
552.3648318680935 CPU and 2084.9902059892192 GPU seconds, including failures,
book retries and the declared 60-second CPU bookkeeping allowance. The campaign
remainder is 74427.31381860161 CPU and 75209.49884549249 GPU seconds.

At the observed dynamic cost, 384 fits still imply 36.97 Gaussian or 30.45
beta-binomial GPU hours, each exceeding the 20.89 remaining GPU hours. M31
therefore reviews an affordable statistical design before further full-fit
calibration; it must not weaken or relabel the original coverage question.
