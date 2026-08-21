# NeuTra target-specific curriculum search plan (2026-08-15)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Can the order and activation of NeuTra transport parameter groups be selected by a reproducible, target-specific search procedure rather than by an inherited fixed curriculum? |
| Candidate mechanism | A bounded replicated probe search over adapter-supplied variable groups, followed by a frozen fresh-data training run and the established downstream predictive-distribution audit. |
| Baseline ladder | Naive cold joint training; tuned cold joint training; the existing fixed staged recipe as a historical/context arm; and the searched curriculum. The historical arm is not silently treated as a current default. |
| Primary selection quantity | Replicated held-out reverse-KL improvement per optimizer update from a common parent checkpoint, summarized by a conservative one-sided uncertainty bound. |
| Primary promotion criterion | The frozen searched protocol passes the untouched downstream predictive-output distribution equivalence test for the target model. |
| Probe vetoes | Nonfinite loss/gradient/state, invalid group/prerequisite graph, changed parent checkpoint across candidates, scalar/non-batched target path, unequal probe budget, or leaked audit data. |
| Final-run vetoes | Any declared predictive-equivalence failure, nonfinite artifact, invalid simulator, missing fresh-data partition, or failed training/inference invariant. |
| Explanatory diagnostics | Probe loss traces, improvement per update, clipping, LR choices, seed dispersion, beam membership, and final proposal moments/ESS. |
| Must not conclude | A probe winner is not a correct posterior, a better method, or a universal curriculum. A passing final protocol is target-specific and does not establish SSL-LSTM transfer until the SSL-LSTM audit itself passes. |

## Why this procedure is needed

The low-level staged controller is target-agnostic, but its group order and
phase budgets are not. The previous fixed recipe passed the reverse funnel,
failed the correlated Gaussian while cold joint passed, and remained unstable
on banana. That is evidence against transferring the recipe, not against a
generic controller. The missing layer is a reproducible method for choosing a
target-specific curriculum.

## Separation of roles

The implementation is split into three layers:

1. **Adapter layer:** the target/model adapter declares transport variables as
   named, non-overlapping groups, prerequisite edges, and any legal joint group.
   The adapter may depend on the model. It must not declare an order as a fact.
2. **Search layer:** the generic search engine probes eligible group additions,
   aggregates replicated evidence, and returns a bounded final beam. It knows no
   target formula, coordinate index, or model name. Its representative sequence
   is execution ordering only.
3. **Protocol-selection layer:** every retained terminal sequence is trained to
   the same total budget on fresh protocol-selection data and common replicate
   seeds. A paired uncertainty-set rule nominates one frozen protocol.
4. **Training/audit layer:** the nominated sequence is converted into stage
   specifications, retrained from a fresh initialization and fresh training
   partition, then evaluated on the untouched downstream predictive-output
   distribution test. Search artifacts cannot directly promote a transport.

## Data and seed partitions

Every target receives disjoint deterministic partitions:

- probe-training batches used only for optimizer updates;
- probe-selection batches used only for probe loss and curriculum selection;
- final-training batches used only after the curriculum is frozen;
- final predictive-audit simulator paths reserved for the scientific claim.

The same probe batches and parent checkpoint are used across competing next-group
probes at a given search node (common random numbers). Replicate seeds are
independent across nodes and routes. A final audit path is never passed to a
probe callback or used to tune order, LR, phase length, or architecture.

## Probe search protocol

For a parent active-group sequence (A), let (C(A)) be the eligible groups
whose prerequisites are contained in (A) and which are not already active.
For every (g \in C(A)):

1. restore the identical parent model checkpoint and optimizer policy;
2. run the same declared probe update budget (B_p), LR grid, batch size, and
   clipping policy;
3. repeat for the predeclared replicate seeds;
4. record incoming selection loss (L_0), best selection loss (L_{g,r}),
   executed updates (u_{g,r}), finite/veto status, and parent-state hash; and
5. compute (d_{g,r}=(L_0-L_{g,r})/u_{g,r}).

A candidate is probe-eligible only when every required replicate is finite, all
replicates share the parent-state hash, and the update budget is exact. For
replicate count (n\ge2), the search computes

\[
\bar d_g = n^{-1}\sum_r d_{g,r}, \qquad
\mathrm{LCB}_g = \bar d_g - c\,s_g/\sqrt n,
\]

where (c) is fixed before the run. A group passes the probe gate only when
`LCB_g >= minimum_improvement_per_update`. The LCB is a nomination rule, not a
claim of superiority. If several groups pass, the beam retains up to `W`
sequences; ties are resolved lexicographically only for deterministic artifact
ordering.

The search expands a bounded beam to depth `D`, with a total probe-update cap.
It stops at a node when no eligible group passes, when the depth cap is reached,
or when the probe budget is exhausted. It must preserve every accepted sequence
and the complete final beam. A deterministic representative may be named only
for execution ordering; it is not the scientifically selected protocol. A
cold-joint candidate is evaluated at every target as a reference; it is not
forced through the group-addition search.

## Full-protocol selection tournament

Local probe improvements from different parent nodes are not directly
comparable enough to select a final curriculum. After beam search, every final
beam sequence, the tuned cold-joint baseline, and any explicitly retained
historical comparator are trained from common fresh initializations for the
same total optimizer-update ceiling. They use a disjoint protocol-selection
partition and at least the declared replicate count.

Let (R_{k,r}) be the terminal held-out loss for protocol (k) and replicate
(r). The empirically lowest-mean protocol is a reference only. For every
protocol, compute paired losses (R_{k,r}-R_{*,r}) and a one-sided upper
uncertainty bound. Preserve the set whose upper bound is below the predeclared
practical loss tolerance. Select the lowest-complexity protocol inside that
uncertainty set, with deterministic name order only as a final tie breaker.
This remains a nomination step: its winner must be frozen and rerun on fresh
training data before the untouched predictive audit.

## Protocol selection and final run

The search and protocol-selection artifacts freeze:

- group sequence and prerequisites;
- probe and final budgets;
- LR candidate grid and schedule;
- optimizer state policy;
- batch and seed partition identities;
- beam width/depth and uncertainty constants; and
- parent/checkpoint identities and the complete terminal uncertainty set.

The final searched protocol is retrained from a fresh initialization using a
fresh final-training partition. The cold-joint baseline receives the same final
optimizer-update ceiling and its own LR tuning. The historical fixed curriculum
is retained only as context unless it is explicitly included as a fresh arm.

The final scientific gate is the established predictive-output distribution
equivalence procedure: freeze the selected parameter representative required by
the target protocol, simulate independent paths from the estimated and true
parameter simulators, and compare the predictive distributions at the declared
horizons. Training loss, probe LCB, importance ESS, and output moments are not
substitutes for that gate.

## Search configuration hypotheses

These are campaign hypotheses, not repository defaults:

| Choice | Provenance | Initial value | Failure mode | Diagnostic |
|---|---|---:|---|---|
| Probe updates (B_p) | Derived from prior 100-update low-dimensional phases | 100 | Too short to distinguish groups | Probe improvement curves and terminal slope |
| Probe replicates | Minimum for a dispersion estimate | 4 | Still noisy for complex targets | Replicate SD and seed disagreement |
| Beam width (W) | Bounded combinatorial search | 2 | Misses a useful branch | Preserve all probe summaries and compare terminal arms |
| Search depth (D) | Bounded group activation count | 4 | Stops before full transport | Depth-stop reason and cold baseline |
| Critical value (c) | Reviewed statistical convention | 2.0 | Too conservative/liberal for small n | Sensitivity arm only if predeclared |
| Minimum improvement per update | Target-scale calibration can nominate it, never the audit | target-specific calibration artifact | Proxy threshold may reject useful groups | Repeat with fresh final run and predictive gate |
| Probe budget cap | Compute bound | 32 group-probes × 4 seeds × 100 updates | Search too narrow | Explicit budget-exhaustion status |

The minimum-improvement threshold must be calibrated before the claim run from
the target's probe-selection loss scale or declared as a sensitivity grid. It
must not be adjusted after seeing final predictive-audit outcomes.

## Skeptical plan audit

| Risk | Audit disposition |
|---|---|
| Search is just another ad hoc order | Repaired: order is produced by a declared prerequisite-constrained replicated search. |
| Training loss becomes a correctness gate | Vetoed: probe loss nominates; untouched predictive equivalence promotes. |
| Probe candidates get unequal budgets | Vetoed: common parent, batches, LR grid, updates, and replicate count are required. |
| Selection data leak into final audit | Vetoed: probe-selection, final-training, and predictive-audit partitions are distinct. |
| One noisy seed determines order | Vetoed: four replicates and an uncertainty-bound gate; beam preserves viable alternatives. |
| Beam width silently ranks methods | Vetoed: viable-set preservation is required; deterministic tie order is bookkeeping only. |
| Prerequisites form an invalid graph | Vetoed: duplicate names, unknown prerequisites, cycles, and self-edges fail closed. |
| Search budget favors complex curricula | Vetoed: equal probe budgets and a total cap; final arms share the same update ceiling. |
| Probe improvements are not comparable across parents | Repaired: local scores prune only within the search; terminal protocols enter a separate equal-budget paired tournament. |
| SSL-LSTM group library is guessed | Vetoed: groups must be declared by an inspected SSL-LSTM adapter and documented with parameter ownership. |
| A passing search artifact is promoted directly | Vetoed: only the fresh final run plus predictive equivalence can support a target-specific candidate. |

### Review verdict

The plan is scientifically executable after the search-policy mechanics tests
pass. The search output remains a protocol nomination artifact. It cannot by
itself establish posterior correctness, HMC readiness, or transfer to SSL-LSTM.

## Testing before campaign execution

Focused tests must cover:

- prerequisite graph validation, including cycles and unknown groups;
- equal-budget and common-parent enforcement;
- finite/veto rejection;
- uncertainty-bound thresholding;
- deterministic beam expansion and viable-sequence preservation;
- cross-candidate common-parent evidence by replicate;
- depth and total-budget stop conditions;
- no-candidate terminal status; and
- paired full-protocol uncertainty-set selection; and
- serialization-ready result fields with no NumPy or target-specific code.

Only after those tests pass should a target-specific canary be planned. The
first scientific campaign should use the Gaussian and banana controls before
touching SSL-LSTM, with a separate reviewed SSL-LSTM adapter and predictive
audit plan required afterward.
