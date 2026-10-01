# Native geometry pilot and cloud evaluation

Continue from pushed 3f3f07ee within the unchanged campaign caps and GPU3 gate.
Question: can pilot point construction, ordered exact target evaluation,
curvature accumulation, eigensystem/QR and completed candidate records execute
in one XLA program without changing the original basis or callback order?
Compare isolated 3582b4ac and current checkpoint on identical frozen normalized
directions, center, scale, step and analytical callback. Scalar pilots retain
interleaved plus/minus order; batched pilots retain all-plus then all-minus order.

Direction generation, zero-direction removal/normalization, train/holdout
partition preparation and the full initializer remain outside this boundary.
State those omissions explicitly rather than claiming a compiled whole pilot
or whole initializer. The direction extent is already fixed by preparation.
No RNG or stream identity changes; frozen clouds are diagnostic authorities.
The new program must still support zero retained directions, no positive
curvature, rank zero, partial nonfinite outputs, and callback construction/
conversion failures. The scalar lane is an initializer reference/evaluation
lane, not batch-native training. The batch lane invokes one actual batched
target and does not row-map a scalar implementation.

Reuse the current native accumulation body and refined symmetric eigensystem;
preserve the original central-difference arithmetic, ordered sketch sum,
positive-curvature predicate, descending eigenvalue selection, thin QR and
identity fallback. Compare every basis element, eigenvalue, curvature summary,
candidate value/score/position/index/eligibility and callback count at unchanged
1e-10 numerical tolerance with exact discrete fields. Eigenvector sign or
degenerate-subspace mismatches are failures to inspect, not an implicit waiver.
Preserve raw original/current/candidate records. Graph references explicitly
disable XLA. Default candidate JIT stays true with stable signatures.

Skeptical review: original scalar pilot candidates retain nonfinite raw scores
and values even when ineligible; original batched cloud evaluation replaces an
invalid row's value and score with NaNs. Preserve that difference and exclude
either from curvature. A finite score paired with a NaN value is ineligible.
Do not count Python tracing as a target call. Use device-colocated int64 resource
counters and record exact points. Unsafe assertion/Python callback graphs must
fail at construction. Preserve pure-resource updates and prohibit eager retry.

Run smallest D3 original comparison first, then D1/D3/D5 scalar/batch boundaries,
changed inputs/resources, HLO/one-trace and callback lifetime checks. Fresh
matched D3/D5 costs and affected public consumers remain required before public
wiring; the existing warm/cold/host/device investigation triggers apply. Use the
same bounded driver, at most three attempts per unchanged job, 120/300-second
workers and versioned numbered artifacts. One numerical worker; frozen source
during workers/matrices. Contention, invalid provenance or budget exhaustion
stop the affected launch. No external source/pin, package, environment, threshold
or tolerance changes. No HMC/posterior/scientific or terminal-repair conclusion.
This primary-agent plan review finds no reason to change the numerical target;
GPU qualification and the omitted prefix/outer scope remain explicit vetoes.

02531 passes the first D3 scalar pilot, including every original basis entry,
diagnostic, exact candidate and callback point/count in graph and XLA. Continue
the predeclared dimension/lane/failure matrix. The shared callback scanner was
extracted without changing its validation; renew its focused checks afterward.

02532 passes all 16 rank-zero D1 records. 02533 passes 14 D3 records and exposes
an inherited empty-pilot compilation failure in the frozen 3f3f07ee checkpoint
before candidate evaluation: XLA still tries to compile an impossible empty
index in the zero-trip sketch body. Preserve that failure, but catch attribution
checkpoint exceptions so they cannot prevent comparison to the original
authority. The candidate now uses static empty-shape branches for zero retained
directions, preserving the original identity basis and empty scalar execution;
an empty batched target is still called once, as originally specified. No
public helper correction is claimed until its separate qualification.

02534--02537 pass all 50 original/boundary/runtime-input checks: 16 at each
dimension, two changed-input/HLO checks. Fresh matched cost arms extract the
unchanged original pilot body after direction normalization, at sketch creation;
archive its AST hash and compare its complete basis/diagnostics/candidate records
with the full original helper. Both modes receive the same normalized directions
and build/materialize all candidate records. Run original/graph/XLA at D3/D5 in
both scalar and truly batched lanes, 21 calls plus a changed-input call per
process. Do not compare the complete original helper against a candidate that
omits normalization. The previously stated investigation triggers remain active.

02538 passes seven malformed-batch and resource-change/release checks.
02539--02550 pass all twelve matched original/graph/XLA CPU cost processes,
including complete changed-input records. The descriptive measurements are:

| Lane | Dimension | Original / graph / XLA warm ms | XLA cold s | Extra RSS MiB |
| --- | --- | --- | --- | --- |
| scalar | 3 | 24.178 / 4.139 / 3.178 | .754 | 210.4 |
| scalar | 5 | 25.355 / 4.788 / 3.857 | .791 | 213.6 |
| batch | 3 | 3.174 / 3.470 / 3.208 | .710 | 209.1 |
| batch | 5 | 4.063 / 4.152 / 4.114 | .752 | 210.1 |

The cold-time investigation trigger fires in every XLA arm. Neither the warm
time nor the 256 MiB extra-host trigger fires. Most allocation occurs at first
execution; twenty warm calls add 12--116 KiB RSS. This short observation does
not establish a memory plateau or general leak freedom. Scalar/batch XLA graphs
contain 569/524 nodes respectively at both dimensions. Analysis and input/source
checksums are in `geometry-pilot-scalar-costs-02550.json` and
`geometry-pilot-batch-costs-02550.json` under the shared campaign artifact root.

02551 renews all 21 shared callback checks; 02552 renews 22 existing geometry
checks; 02553 passes all 72 policy/controller checks. Audit 02554 covers 2,983
working Python files, 2,982 parsed and one unchanged vendor-reference parse error.
The guard covers 212 sources with 1,306 exact exceptions and no violations or
stale entries. No numerical-loop exception was added. Focused Ruff and whitespace
pass. An initial audit command omitted `--device CPU` and stopped in the driver's
default GPU preflight before launching any worker; the successful audit explicitly
hides GPUs. No scientific result is attributed to that blocked launch.

| Decision | Primary criterion | Vetoes / uncertainty | Next action | Unsupported conclusion |
| --- | --- | --- | --- | --- |
| Retain internal pilot checkpoint | All original/boundary/runtime CPU records pass | GPU3 occupied; cold-time overhead; omitted normalization and outer initializer | Compile preparation and qualify GPU when idle | Public GPU or terminal repair readiness |

Review finds the original extraction and candidate boundaries matched, with no
tolerance or callback-order waiver. The weakest evidence is one fresh process
per cost arm and no long pilot reuse measurement; terminal repeated whole-endpoint
costs remain necessary. No public helper is wired to this candidate yet.
