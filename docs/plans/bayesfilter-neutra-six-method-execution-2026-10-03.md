# Six-method NeuTra study: execution and approval setup

October 4 status: the bounded mechanics/pricing phase and terminal controller
repair are complete. See [the result and remaining scientific gaps](bayesfilter-neutra-six-method-results-2026-10-04.md)
and [the recovery memo](bayesfilter-neutra-six-method-reset-2026-10-04.md).
This is not completion of the generic scientific study.

The owner authorized implementation, plan review and execution on October 3,
2026, including a narrow persistent command allowance. The controlling
scientific design is [the generic plan](bayesfilter-neutra-generic-recovery-and-transfer-plan-2026-10-03.md).
This note freezes the initial implementation/validation/pricing allocation;
it does not pretend that the unpriced 636-trial confirmation design fits it.

## Recovered state and skeptical review

The active checkout is `preserve/shared-main-before-fab-20260926` at
`70a6d7e96`, with substantial unrelated changes. Preserve them. The shared
September 29 ledger has 47,639.23 GPU process seconds and 134,777.06 CPU core
seconds conservatively available (13.233 and 37.438 hours). The historical
October 2 allocation is a suballocation, not an additional grant. NVIDIA's
trusted probe finds devices 0 and 2 busy and device 1 free. Initially use GPU
1 only; do not kill or displace another campaign.

Material findings before implementation:

1. FAB exists in the committed canonical worktree, not in this active source
   tree. Recover the existing port with its source/license and test its actual
   shared-transport interface; do not invent a replacement and call it a port.
2. The active transport core is FP64, despite the separate precision worktree.
   Initial mathematics/mechanics and pricing are explicitly GPU/FP64 reference
   experiments. They cannot certify FP32/TF32 performance or default readiness.
   No global dtype mutation or unrelated dirty-file overwrite is allowed.
3. Existing SMC mutates only after resampling. Turning resampling off would
   produce sequential importance weighting without AIS mutation. Add a fixed
   schedule option with mutation at every stage and retain old behavior only
   for existing configurations. Pair AIS/SMC through that common valid path.
4. Existing target names expose fixed centers, and downstream helpers use them.
   Introduce a parameterized density/score interface separated from evaluator
   truth. Existing fixed-target results do not constitute generalization.
5. FAB's auxiliary density needs a tail argument for the actual map/target.
   An undefined objective is a mathematical outcome, not a reason to run a
   finite-weight canary and promote it. Preserve source replay and no-replay
   profiles separately and inspect their gradients.
6. Every cell must remain visible after failure. A successful SMC cell cannot
   end the matrix. All post-training score probes are explanatory; correctness
   and posterior promotion require the declared independent downstream checks.

These findings invalidate launching the historical master unchanged. The
revised implementation phase passes skeptical review because it first tests
the missing interfaces, invariants and actual consumer paths and prices them;
it does not claim method success from those checks.

## Evidence contract and bounded first allocation

Question: do the six declared method consumers compute their intended
quantities on exact controls, and what does the complete assessment cost?
Comparators are analytic Gaussian identities, fixed-input author fixtures,
paired AIS/SMC and identity-transport SMC controls. The primary engineering
criterion is agreement with these quantities, complete state restoration and
correct failure accounting. Wrong target/weights/derivatives, false source
identity, reference leakage and invalid GPU memory policy veto dependent work.
Runtime, loss, acceptance and short-run moments are descriptive diagnostics.
No posterior convergence, generalization, method ranking, production readiness
or FP32/TF32 claim follows from this phase.

Initial ceiling: 7,200 GPU process seconds and 14,400 CPU core seconds, inside
the shared remaining balance. These are convenience ceilings for bounded
mechanics/pricing, not tuned scientific parameters. Every attempt is charged,
including failed compilation and retries. Each distinct job gets at most one
localized infrastructure retry after repair, with a fresh directory. A source
change creates a fresh attempt and recorded source snapshot. CPU smoke tests
hide GPUs and cannot support learned-map quality claims.

Output root: `docs/plans/artifacts/neutra-six-method-2026-10-03/campaign-r1`.
Preserve source, configuration, command, environment, seed, GPU memory policy,
per-attempt CPU/wall/GPU time, result and next-phase decision. The existing
shared campaign lock and ledger remain authoritative. Development, simple-case
confirmation and generalization are separately priced before their launch;
insufficient budget changes the reported scope, never the scientific claim.

## Fixed commands and approvals

Use exactly `bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_six_method_campaign.sh`
followed by one fixed verb: `check`, `preflight`, `run`, `resume` or `status`.
The wrapper sets memory growth before TensorFlow import, uses the fixed tfgpu
interpreter and bounded controller, and accepts no arbitrary command argument.
Approve those commands together through a dedicated rules file installed by
a fixed installer. Do not add general Python, Bash, sudo or arbitrary-script
allowances. No sandbox/reviewer/gateway setting is weakened.

The read-only route doctor reports local/session readiness. Its local allow
decision cannot override the managed platform. Official documentation retrieval
returned HTTP Forbidden; actual local rule parsing and exact direct/shell argv
checks will therefore be used to verify the installed behavior, without making
claims about unavailable documentation. If a platform approval is still
required, preserve its exact reason rather than changing command packaging to
evade it.

## Execution order and refresh

1. Recover the FAB implementation and construct generic targets/evaluator.
2. Repair AIS/SMC stage semantics and add focused mathematical/control tests.
3. Wire all six methods to their actual shared implementations, preserving
   native objectives and the posterior-teacher distinction.
4. Validate fixed command rules and CPU mechanics; trusted memory-growth/XLA
   preflight; bounded per-method GPU controls and pricing.
5. Reconcile the complete matrix and resource ledger. Repair localized failures
   within the allocation. Freeze the next executable phase only when the
   artifacts answer its question and its measured cost fits the remaining
   authorized budget. Otherwise record the precise blocker and priced need.

Every phase refreshes `state.json`, `matrix.json`, `next-phase.json` and a
result note. The master reports unexecuted/inapplicable/dependent-failure
cells explicitly. A completed engineering phase is not the completed study.

## FAB applicability finding for the quadratic-shear fixture

The target is the exact mixture transformed by
`x_1=u_1, x_2=u_2+kappa*(u_1^2-a)`, with nonzero kappa. Every finite canonical
IAF stage has bounded conditional log scale `b+c*tanh(h/c)` and an ELU
shift network with finite weights. ELU is globally 1-Lipschitz, so the shift
has at most linear growth. Consequently the finite composition satisfies
`||T(z)|| <= A||z||+B` for finite A,B, and its absolute determinant is bounded
below by a positive constant d_min (sum the finite scale lower bounds).
Therefore, for x=T(z),

\[
q(x)\le (2\pi)^{-D/2}d_{\min}^{-1}
\exp\{-((\|x\|-B)_+)^2/(2A^2)\}.
\]

In the positive-area tube `x_1=t`, `x_2=kappa*(t^2-a)+u`, `|u|<=1`,
the shear Jacobian is one. One positive-weight mixture component bounds
gamma(x)^2 below by `C exp(-c_2 t^2-c_1 |t|)` for finite positive constants,
while `||x||` grows quadratically in |t|. Thus gamma(x)^2/q(x) is bounded
below, eventually, by `C' exp(c_4 t^4-c'_2 t^2-c'_1 |t|)` with c_4>0.
Its integral over this tube diverges. This rejects the alpha-two auxiliary
density for this exact warped target and this bounded-scale ELU IAF family;
it is stronger than a failed numerical canary and is not a theorem against
posterior AIS/SMC or FAB with another eligible proposal family.

The unwarped-mixture nonlinear-IAF tail question is distinct and remains
unchecked. A Gaussian proposal q=N(m,C) is eligible exactly when all diagonal
mixture terms have positive precision `2/v_k I-C^{-1}`. The initial FAB
mechanics/pricing control therefore uses a deliberately zero-network affine
fixture on a unit Gaussian (q variance initially 4) and rechecks that exact
condition before each update. Zero weights make its hidden-network gradients
zero, so only affine biases move. This is an objective/port control, not a
canonical initialization or a viable nonlinear-training result. It must not
be counted as completing FAB's scientific-study row.

The initial control settings (256 particles, eight fixed intervals, four
MALA moves, batch 64, 16 native updates, 32 student updates, scale 4, step .1,
learning rate .001 and clipping ceiling 1000) are small engineering hypotheses
for graph/operation pricing, not tuned values. AFT uses two stage updates and
CRAFT two passes for this purpose. Use 4,096 independent evaluator samples
only to exercise the evaluator and the required 1,000-point post-training
probe. No short-chain accuracy or learned-map claim is allowed. The fixed
Gaussian price control is isotropic; parity against the existing correlated
Gaussian is tested independently. Full budgets cannot be forecast by assuming
these update counts train the nonlinear maps adequately.

## Terminal review and controller repair, October 4

Recovery found all 18 mechanics/pricing cells recorded: 16 completed controls
and two FAB applicability outcomes. This finishes the bounded numerical
execution, not the scientific study. The next authorized work is the terminal
artifact/source/cost audit and a focused repair of the master itself. The
installed five-verb allowance remains sufficient; no broader command,
reviewer, gateway or sandbox change is needed.

Skeptical review found three engineering defects. First, creating a new source
snapshot on every resume made unrelated edits repeat completed controls.
Second, CPU tests ran in the live checkout while their manifests identified a
snapshot; those historical test results cannot establish frozen-source test
coverage. Third, the ledger included child shutdown CPU time while GPU-worker
manifests stopped earlier, without explicitly identifying the difference.
The historical artifacts will be preserved and these limits recorded, not
rewritten into stronger evidence. The first control attempt also repeated a
shared post-training graph-construction failure across 16 cells; that failure
was repaired before the second attempt.

Repair the master to preserve terminal progress, run new checks from a complete
test snapshot (including test configuration and the independent author
fixture), save both worker and whole-process accounting in new manifests, and
audit all selected controls before reporting engineering completion. A resume
of a terminal matrix must audit and report without starting GPU work. Missing
artifacts, changed snapshots, duplicated/missing ledger charges, false
promotion fields and failed prerequisites reported as successes are failures
of this audit. Historical live-tree tests are an explicit limitation, repaired
prospectively by a fresh frozen-source check. Source-faithfulness and teacher
qualification remain separate unresolved scientific tasks.

Validation uses the existing exact `check` command for a fresh CPU mechanics
suite plus controller regressions, then the exact `resume` and `status`
commands. Controller tests exercise terminal resume after unrelated edits,
failed or incomplete matrices, snapshot execution, accounting reconciliation
and rejection of extra launcher arguments. They do not import an accelerator
framework. The check retains the existing per-attempt 600-second wall cap and
two-core CPU allowance, charged inside the unchanged 7,200 GPU-process-second /
14,400 CPU-core-second allocation. These inherited engineering ceilings are
limits, not estimates of scientific convergence. No new GPU numerical run is
needed to verify this controller repair.

This revised repair passes skeptical self-review: it uses preserved attempt
artifacts as the baseline, checks actual execution provenance, does not treat
short-run diagnostics as scientific success, and has explicit resource and
failure bounds. The terminal result will include all 18 outcomes, the failed
attempts and repairs, the source/controller gap inventory, and measured costs.
Pricing the full scientific phase remains incomplete until actual calibrated
teachers, adequate training and fresh HMC qualification have been measured;
extrapolating 32 debug updates would give a misleading forecast.
