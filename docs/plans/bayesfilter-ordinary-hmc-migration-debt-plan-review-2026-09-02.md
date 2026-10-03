# Skeptical Review Of The Ordinary HMC Migration-Debt Plan

Date: 2026-09-02

Reviewed plan:
`docs/plans/bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md`

Review type: pre-execution skeptical developer audit. This is a review of a
static trace and repair plan, not approval to run HMC or to change a numerical
default.

## Verdict

**PASS FOR STATIC DOCUMENTATION, SOURCE-TRACE, AND CLASSIFICATION WORK.**

**REVISE BEFORE NUMERICAL EXECUTION OR ORDINARY-POLICY PROMOTION.** The plan
correctly identifies an authority/implementation mismatch and several
downstream migration debts, and it gives bounded checks that can answer those
questions without a chain run. It must not be treated as an approved choice
of measured joint tuning, per-L adaptation, or any other ordinary numerical
policy. The owner decision and a new evidence contract remain required.

## Findings First

### 1. Baseline risk: the plan could mistake a capability description for the
executable baseline

The review checked the baseline against source rather than against the prose:

* `tune_hmc_kernel` dispatches on config type in
  `bayesfilter/inference/hmc_tuning_dispatch.py:29-84`.
* The ordinary config defaults to the operational fixed-trajectory algorithm
  in `hmc_kernel_tuning.py:6401-6460`.
* An operational warm-up result selects the old fixed-mass/step stage in
  `hmc_kernel_tuning.py:10348-10392`; that stage invokes the three-replicate
  floor/anchor/double selector.
* The newer joint grid is reachable in the alternate stage path and is named
  `joint_l_epsilon_grid_fixed_mass_hmc`, but the route contract calls it
  legacy/non-promoting while its payload says `promoted_default=True`.
* The ordinary module header (`hmc_kernel_tuning.py:1-15`) calls the joint grid
  promoted even though the default branch selects the operational selector.
* The implicit `standard` preset is labelled diagnostic-only and uses a
  non-serious budget even though the enclosing function is registry-active.

The plan repairs this by requiring a resolved policy record before comparing
or promoting anything. This is necessary. A broad-grid or fixed-transport
result is not a valid baseline merely because it has a richer candidate set.

### 2. Proxy-metric risk: acceptance, R-hat, ESS, and efficiency could be
silently promoted

The plan explicitly classifies acceptance, energy, path return, ESJD/gradient
cost, R-hat, ESS, and runtime as explanatory until a separate uncertainty
design promotes one. It also records the existing four-chain/four-block
acceptance screen and its 90 percent t calculation as a screen, not
convergence evidence. This matches the source policy in
`bayesfilter/inference/hmc_verification.py:452-580` and the repository
statistical rules.

The future numerical phase still needs a predeclared primary criterion and
uncertainty method. The plan correctly leaves that as an owner decision. No
claim of "best," "improved," or "superior" is licensed by the static checks.

### 3. Stop conditions and repair triggers are present, but campaign numbers
must not be invented during implementation

The plan has hard static vetoes, continuation vetoes, localized repair
triggers, versioned artifact requirements, and a boundary between static work
and a future campaign. It does not invent a new run count, wall-time budget,
chain length, or threshold. Existing values are labelled source observations.

Before Phase 8, the implementer must add a concrete total compute/attempt
budget and stop conditions to a new experiment plan. Reusing the old "three
replications" or "64 observations" as a scientific power claim would fail this
review.

### 4. Comparison fairness is addressed at the right level

The plan binds target, coordinates, mass, start bank, seeds, dtype/backend,
and adaptation settings to the artifact and calls for disjoint calibration,
selection, verification, and claim data. This prevents a measured grid from
being compared with a frozen-epsilon selector under different geometry or
data. It also keeps the fixed-transport route separate.

The implementation phase must make these bindings executable, not just prose:
the replay test must fail when any bound field changes. A source signature
that is merely caller-stamped is insufficient.

### 5. Hidden defaults and overloaded dispatch remain the highest-risk defect

The plan names the hidden selector, replication count, acceptance settings,
grid constants, TFP defaults, XLA divergence, seed conventions, and mass
sequencing. It also requires a config-variant and policy ID in every result.
This directly addresses the reason downstream agents repeatedly choose the
wrong interface.

One remaining decision is whether to split public names or retain a typed
dispatcher. The plan correctly treats this as an owner decision. If one name
is retained, the type and resolved policy must be visible in the return schema
and an authority request must fail closed for mechanics-only variants. If
names are split, a compatibility wrapper must not silently upgrade evidence.

The same rule applies to presets: `standard` and `serious` are not interchangeable
just because they share the function name. The plan now requires the preset role
and budget class to be part of the authority decision.

### 6. Stale context and naming are explicitly red-flagged

The plan does not trust the words "legacy," "promoted_default," "joint," or
"tuning authority" in isolation. It asks for source-line reconciliation and
preserves the prior audit and historical artifacts. This is important because
the route contract and stage payload currently use conflicting labels.

The implementation must update names and generated tables together, then run
the renderer check. A prose-only change would leave the same migration debt.

### 7. Environment and dirty-worktree risks are bounded

The plan records the observed revision and dirty-worktree condition, limits
the first phase to read-only/static commands, and requires small commits that
preserve unrelated edits. No GPU, package, network, or destructive command is
needed. A future numerical plan must record the actual environment and
hardware, rather than inheriting the static-phase environment silently.

### 8. Artifact adequacy is mostly sufficient, with one required refinement

The proposed artifacts answer the static question: a trace, policy schema,
consumer inventory, route tests, documentation tests, and a final result note.
For numerical work, the plan correctly requires policy lineage, replay fields,
source closure, seeds, and versioned output roots.

The refinement is to make the top-level tuning result expose the *resolved*
branch, not only the caller config or a nested stage payload. Otherwise a
consumer can still mistake an operational handoff for the joint-grid result.
Add a construction-only assertion that the result's authority role agrees
with the route registry and policy record.

### 9. Backend default is a policy mismatch unless explicitly exempted

The ordinary public config defaults `use_xla=False` in
`bayesfilter/inference/hmc_kernel_tuning.py:6401-6460`, and that value is
forwarded through the public route. The repository policy requires XLA JIT as
the default for algorithmic TensorFlow paths; non-XLA is permitted only when
the run is explicitly classified as a reference, smoke, debugging, or
reviewed exception. The current active function/preset does not expose an
exception identity. This is a confirmed policy-boundary defect, not a runtime
performance result. The plan must require either an XLA-on default after
compatibility checks or an explicit, scope-bound exception record and guard.

### 10. A compatibility definition can still mislead static consumers

The monolith defines a second public-looking `tune_hmc_kernel` delegate at
`hmc_kernel_tuning.py:14461-14500`, while lazy exports and the registry point
to `hmc_tuning_dispatch.py:29-84`. The delegate currently forwards to the same
dispatcher, so no second numerical policy is established by this fact alone.
It does create a confirmed import-path and annotation ambiguity, and the
current route inventory excludes compatibility aliases. The repair should keep
one canonical public definition, mark the delegate explicitly deprecated or
private, and test that direct imports cannot create an unregistered authority
route.

## Plan Repairs Applied Before Approval

The plan was revised during this review to include:

* an explicit current call graph and alternate branch;
* a distinction between function-level registry authority and policy-level
  authority;
* the TensorFlow-native mechanics branch under the same public name;
* the legacy/non-promoting route naming conflict;
* direct downstream diagnostic/private imports and raw-runner classification;
* a second consumer/bypass scanner rather than relying on the existing route
  inventory;
* a policy/assumption ledger with provenance for every observed number;
* explicit separation of `standard`/diagnostic preset roles from serious
  authority;
* replay invalidation for mass, target, coordinates, backend, dtype, and start
  bank changes;
* preservation of the already repaired fixed-transport measured-grid route;
* a separate NumPy/export cleanup phase; and
* the XLA-default/exception boundary and duplicate compatibility definition;
* a strict boundary saying that no HMC or numerical promotion follows from
  this document.

## Required Review Questions Before Implementation

An independent reviewer must answer these questions from source anchors:

1. Does the default ordinary config always reach the operational selector, or
   can another default/runner path select the joint grid?
2. Does the top-level result and replay schema bind the resolved algorithm,
   config variant, and policy version, or is a new field required?
3. Can a legacy/non-promoting algorithm be reached through an authority route?
4. Which MacroFinance and `dsge_hmc` files are claim-adjacent versus
   mechanics-only or historical?
5. Which NumPy imports are on an admitted runtime path, and can they be moved
   without changing the target?
6. Are the guide, generated table, capability registry, and executable
   defaults mutually consistent after the current dirty edits are separated?
7. Does a passed `standard` preset result have the authority the caller is
   likely to infer, or must claim-adjacent callers require `serious` explicitly?
8. Is `use_xla=False` a documented, scope-bound exception or an unreviewed
   violation of the repository XLA-default policy?
9. Can the compatibility definition in `hmc_kernel_tuning.py` be imported or
   scanned as an independent public route despite delegating to the dispatcher?

If any answer is "not checked," implementation should remain in Phase 0/1.

## Acceptance Table

| Review item | Status | Evidence/next action |
|---|---|---|
| Wrong baseline avoided | Pass for static phase | Source trace names operational default; owner choice still open |
| Proxy metrics separated | Pass | Roles and nonclaims are explicit; add uncertainty design later |
| Stop/repair rules | Pass with condition | Add campaign budget only in future numerical plan |
| Fair comparison | Pass with condition | Enforce replay signatures in code |
| Hidden defaults | Pass for inventory | Bind resolved policy and config variant in result |
| Preset authority | Open repair | Require an explicit role/preset for claim-adjacent consumers |
| Stale docs/registry | Open repair | Reconcile legacy and `promoted_default` labels |
| Environment mismatch | Pass for static phase | Record actual environment for future runs |
| Artifact answers question | Pass with refinement | Add top-level resolved-policy field and construction test |
| External consumer scope | Open audit | Complete per-file role ledger before migration |
| Backend default policy | Open repair | Resolve XLA-on default versus explicit exception and guard it |
| Duplicate public definition | Open repair | Retain one canonical import path and test the alias boundary |

## Final Review Statement

The plan is coherent and safe to execute only as a read-only source,
documentation, and classification exercise. It is not yet a numerical
experiment plan and cannot authorize a default change. The next action is the
bounded Claude audit in the accompanying handoff memo, followed by owner
resolution of the ordinary policy and public API boundary.

`VERDICT: PASS_FOR_STATIC_AUDIT; REVISE_BEFORE_NUMERICAL_EXECUTION`
