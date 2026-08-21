# SSL-LSTM q=20 NeuTra global-mixing repair plan (2026-08-19)

Status: `PLAN_REVIEWED_READY_FOR_CANARY`

This plan supersedes the upstream-archive gate in
`bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md`.  That gate was
circular: an independently converged global posterior archive would already
solve the sampling problem for which NeuTra is being developed.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can a target-trained NeuTra transport make the exact SSL-LSTM q=20 pullback target globally traversable by one fixed-HMC kernel, without requiring a global posterior archive as training input? |
| Candidate mechanism | Target-weighted forward-KL training from a full-support, target-query-built proposal replay cloud, followed by exact fixed HMC on the transformed target. |
| Historical baseline | The old single-region reverse-KL transport and the failed small three-mode transport. They are local/mode-seeking baselines, not posterior authorities. |
| Positive control | The reviewed three-mode weighted forward-KL transport, where one exact pullback HMC target mixed among all components and recovered component masses. This is a mechanics precedent, not a transferred SSL default. |
| Expected failure modes | Replay weights may be too concentrated; the proposal may miss a material mode; the learned map may remain highly nonuniform; fixed HMC may retain initialization dependence; target invalidity may occur in a tail. |
| Promotion criterion | After training, a single exact transformed-target HMC protocol passes finite/status gates, modern R-hat and ESS including the mode indicator, and initialization-forgetting/cross-mode gates for every chain; only then may its retained draws support predictive testing. |
| Promotion veto | Any nonfinite target/transport, invalid status, stale identity, scalar training fallback, proposal-support collapse, chain-specific mode trapping, failed modern R-hat/ESS, or use of conditional mode chains as a pooled posterior. |
| Continuation veto | The target adapter or transport is not exact/bijective, the replay cannot be evaluated finitely, or the canary cannot complete within the bounded compute cap. |
| Explanatory diagnostics | Proposal ESS, weighted NLL, latent means/covariances, acceptance, transition counts, occupancy, and SMC mass comparison. These diagnose the candidate and do not establish posterior correctness alone. |
| Must not be concluded | No exhaustive mode-discovery proof, exact posterior-weight proof from the replay, parameter-identifiability claim, or method/default superiority from a passing candidate. |

## Mathematical correction to the prior plan

Let the learned bijection be `theta = T_phi(z)` and the exact transformed
target be

```text
pi_phi(z) proportional to pi_theta(T_phi(z)) * abs(det J_T_phi(z)).
```

For any fixed reversible, volume-preserving HMC proposal with a Metropolis
correction, `pi_phi` is invariant.  This does **not** make a collection of
mode-restricted chains a posterior sample.  If chain `j` never leaves region
`A_j`, its empirical limit is `pi_phi(. | A_j)`.  Pooling those chains with
equal counts gives the desired target only when the true region probabilities
are equal; otherwise the pooling weights are imposed by the start design, not
learned from the target.  Therefore:

1. starts near each known mode are permitted as an overdispersed diagnostic;
2. they are not a source of mode weights and their conditional draws cannot be
   pooled;
3. promotion requires the same exact HMC kernel to forget the starts and cross
   every material mode; and
4. mode-specific runs that do not mix are a transport failure, not a posterior
   sampler.

If `T_phi` were exact, `pi_phi` would be the chosen simple base law.  NeuTra's
scientific purpose is to make this pullback close enough to that law that one
fixed HMC kernel can explore it.  The downstream gate therefore tests global
mixing of `pi_phi`, rather than merely local movement from each mode.

## Why training does not require a posterior archive

Forward KL is

```text
KL(pi || q_phi) = E_pi[-log q_phi] + constant.
```

For a full-support proposal `r`, target evaluations provide the self-normalized
importance approximation

```text
E_pi[f] approximately sum_i w_i f(theta_i) / sum_i w_i,
 w_i = exp(log-tilde-pi(theta_i) - log-r(theta_i)).
```

This is a finite training estimator, not an independent posterior archive.  It
can be imperfect without biasing the exact HMC target: the frozen transport is
only a coordinate change, and HMC still evaluates the original target and the
Jacobian.  A missed mode can make the coordinate map inefficient, so proposal
coverage remains a training risk and is tested explicitly.

## Evidence contract

### Question and comparator

The question is whether a fresh weighted forward-KL transport yields a globally
mixing exact pullback HMC run.  The comparator is the existing reverse-KL,
single-region transport; no old posterior draws are reused.  The physical
dense-mass run is an optional diagnostic comparator, not an upstream gate.

### Primary criterion

For a fresh transport seed, all retained chains must satisfy the shared
sequential HMC policy (`L >= 2`, no NUTS):

- finite state, target, Jacobian, and target-status receipts;
- modern rank-normalized split and folded R-hat `<= 1.01` for parameters and
  the known-mode indicator;
- declared bulk and tail ESS thresholds, including the mode indicator;
- every chain visits both known sign regions in retained draws and produces
  transitions caused by HMC, not merely by its initialization;
- occupancy and transition diagnostics are consistent across differently
  initialized chains within their reported Monte Carlo uncertainty; and
- warm-up is excluded from all posterior and predictive estimates.

The per-chain sign/region requirement is a coverage veto, not a claim that two
known signs exhaust the posterior.  If it fails, the candidate is rejected and
the conditional chains are not pooled.

### Diagnostics and roles

| Diagnostic | Role | Interpretation |
|---|---|---|
| Replay ESS/max weight | Repair trigger/support diagnostic | Determines whether the proposal gives usable training signal; never a posterior gate |
| Held-out weighted NLL and latent moments | Selection/explanatory | Selects a checkpoint on disjoint data; does not prove HMC validity |
| Inverse/forward/Jacobian parity | Hard engineering veto | Confirms the map and exact transformed target are implemented consistently |
| Parameter and mode-indicator R-hat/ESS | Promotion criterion/veto | Tests convergence of the actual retained HMC state |
| Per-chain visits and HMC transitions across signs | Promotion veto | Detects conditional-chain pooling and initialization trapping |
| SMC interval `[0.405731, 0.536018]` | Explanatory comparator only | Checks broad consistency over the two known proposal-supported regions; not an authority |
| Predictive path tests (`n=1000`, `T=10,20,30,50,100`, separate `alpha=0.01`) | Scientific endpoint after HMC admission | Tests output-law equivalence, not parameter equality or exhaustive posterior correctness |

### Nonclaims

Passing this contract would show that the candidate's exact transformed HMC
run is a viable posterior sampler under the tested finite evidence.  It would
not prove that no third mode exists, that the proposal weights are exact, or
that NeuTra is universally successful.

## Target-specific default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Weighted forward KL | Successful three-mode control and reverse-KL SSL failure | Mass-covering estimator supplies signal to both known regions | Replay support can still omit a mode | Global replay ESS and held-out support audit | Target-specific hypothesis |
| Two local Gaussian components | q=20 target-query MAP/curvature artifacts | Both known stationary regions are represented with exact target queries | Third/narrow mode is absent | Prior-scale multistart and tempered target-query search | Coverage warm start, not authority |
| Defensive mixture scales | Physical direct-IS/AIS weight-tail failures | Adds tails around each queried region without changing the exact target | Too broad a component can waste rows and collapse ESS | Pilot scale ladder and global ESS | Unproven proposal hypothesis |
| `(128,128)`, six-stage dense IAF | Three-mode weighted control | Capacity precedent for separated modes | Target-specific under/overfit | 64x64 and 128x128 canaries on disjoint replay | Warm-start hypothesis |
| Two fresh training seeds | Minimum replication for a learned candidate | Separates a single optimization accident from a repeatable result | Still too few for a success-rate claim | Independent initialization and replay seeds | Replication minimum |
| 2,000 warm-up / 1,000-window recent readiness | Repository NeuTra sequential policy | Prevents fixed short-run promotion | Cost may exceed cap | Canary timing and checkpoint receipts | Reviewed policy |

## Execution phases

### Phase 0: Correctness and anti-pooling tests

Add focused diagnostic tests with a synthetic two-component target:

1. two chains that remain in separate components but have excellent within-mode
   R-hat must fail the global mode-indicator/transition gate;
2. chains initialized in different components that subsequently cross must pass
   the anti-pooling gate subject to ordinary R-hat/ESS; and
3. equal concatenation of conditional component draws must never be labeled a
   posterior archive unless an explicit external weight is supplied (and that
   external weighting remains diagnostic here).

No target or GPU run is promoted from these tests.

### Phase 1: Proposal/replay canary

Use only target-query MAPs, local curvature, and target value/status calls to
build a defensive two-region proposal. Generate disjoint train/selection/audit
replay banks with CPU/XLA workers. Store proposal log density, target value,
status, region label, seed, and worker provenance for every row. Do not call
these rows posterior samples.

The canary stops before training if rows are nonfinite or if all useful weight is
concentrated in a few rows. The exact thresholds are recorded from the pilot;
they are proposal-quality repair triggers, not posterior promotion thresholds.

### Phase 2: GPU/XLA weighted transport training

Train the weighted forward-KL dense IAF on the replay bank, preserving the
leading batch dimension. Use memory growth before TensorFlow initialization,
float64, XLA, and at least two independent seeds. Tune architecture and a
target-specific learning-rate schedule on the selection bank; freeze the
checkpoint before touching the audit bank. The dense-mass physical archive is
not required.

Selection checks include finite loss/gradients, held-out weighted NLL, latent
scale/correlation diagnostics, and inverse/forward/Jacobian parity. These are
nomination and engineering checks; they cannot promote the transport without
the exact transformed HMC run.

### Phase 3: Exact transformed HMC global-mixing canary

For each frozen transport, construct the exact pullback target from the original
SSL-LSTM target plus the transport log Jacobian. Tune a fixed HMC grid with
`L >= 2` using the shared sequential controller. Initialize chains from broad
base draws and from the two mapped MAP neighborhoods solely to test
initialization forgetting. All chains use one common target/kernel policy; no
mode-specific chain is admitted separately.

The canary is successful only if every initialized family reaches both signs,
the mode indicator has finite modern diagnostics, and no chain remains
conditional. A failure triggers transport/proposal repair, never pooling.

### Phase 4: Material HMC and predictive endpoint

Run the predeclared warm-up/retained ladder only after Phase 3. Archive every
warm-up chunk but exclude it from estimates. On admission, draw one parameter
per posterior-predictive path and compare 1,000 generated paths with 1,000
true-parameter paths at each of the five horizons. Run five separate 1% tests;
do not form an omnibus p-value.

## Skeptical plan audit

1. **Circular baseline:** removed. No global posterior archive is an input or
   prerequisite; only target-query replay and exact target evaluations are used.
2. **Invalid conditional pooling:** explicitly vetoed. Starts are diagnostic;
   only a single kernel that mixes across modes can produce retained evidence.
3. **Proxy promotion:** replay loss/ESS and acceptance are not promotion gates;
   transformed HMC convergence and mode-indicator diagnostics are primary.
4. **Unknown mode:** the two-region proposal is labeled a warm start. The plan
   makes no exhaustive-discovery claim and records this as the dominant
   scientific uncertainty.
5. **Target drift:** exact target/Jacobian signatures bind every replay and HMC
   artifact; old seed-B and dense warm-up states are excluded.
6. **Numerical drift:** all target statuses, finite masks, XLA identity, and
   memory-growth receipts are mandatory; `L=1` and NUTS remain forbidden.
7. **False success from initial occupancy:** per-chain transitions and mode
   indicator R-hat/ESS are required, so equal initial mode counts cannot pass.

Audit verdict: `PASS_WITH_EXPLICIT_COVERAGE_LIMIT`. The plan is non-circular and
mathematically aligned with NeuTra's purpose. The remaining unknown-mode risk is
an explicit nonclaim and repair trigger, not hidden authority.

## Planned artifacts and bounded commands

- Plan: this file.
- Anti-pooling tests: `tests/test_ssl_lstm_q20_neutra_global_mixing.py`.
- Replay/training/HMC canaries: new versioned roots under
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/`.
- Material result: a separate result note only after the canary passes.

The first executable action is Phase 0 plus a replay canary. No dense physical
material rerun and no conditional per-mode HMC run is authorized by this plan.
