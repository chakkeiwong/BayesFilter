# Corrected Parameter-Authority Phase 48 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v3.0-independent-proposal-mutation`  
Entry gate: Phase 47 report completed with branch `mh_rejuvenation_does_not_reduce_variability`  
Status: `PASS_V3_0_INDEPENDENT_MH_REPORT_REPAIR_TRIGGERED`  
Local cap: 5400 s

## Question and scope

Does an independent-proposal Metropolis-Hastings rejuvenation kernel reduce
finite support variability when it uses the exact defensive-mixture proposal
already declared in the theta measure? Phase 47 tested only local isotropic
random-walk moves. The new arm can move between the two calibrated proposal
components and the defensive tail, while preserving the same tempered bridge.

The target remains the batch-native q=20 SSL-LSTM target in `theta in R^4`.
The UKF state is internal to the target evaluation. Identity and independent-MH
arms share exact regenerated initial tensors, resampling seeds, schedule,
particle count, proposal geometry, and target/status calls. No NeuTra training,
HMC, whitening, or LEDH route is launched.

## Mathematical kernel

Let `q(theta)` be the normalized defensive-mixture density and `V(theta)` the
q=20 target log density. At annealing level `beta`, the bridge is

`pi_beta(theta) proportional to q(theta)^(1-beta) exp(beta V(theta))`.

For an independent proposal `theta_prime ~ q`, the Metropolis ratio is

`a(theta,theta_prime) = min(1, pi_beta(theta_prime) q(theta) /
  [pi_beta(theta) q(theta_prime)])`.

The implemented log ratio is therefore

`min(0, bridge(theta_prime) + log q(theta)
       - bridge(theta) - log q(theta_prime))`.

At `beta=0`, `pi_0=q`, so every valid candidate is accepted. At later beta,
the ratio is equivalent to

`beta * [(V(theta_prime)-log q(theta_prime))
         -(V(theta)-log q(theta))]`.

Candidates with invalid target/status or non-finite proposal density are
rejected. The proposal density is evaluated in `theta_R4`; no chart Jacobian
or internal-state density is inserted.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Does nonlocal invariant mutation reduce finite support variability? |
| Mechanism | Independent MH with candidate rows sampled from the exact defensive mixture `q`. |
| Comparator | Identity mutation with identical initial clouds, resampling seeds, and target. |
| Expected failure | Acceptance collapses at high beta, support remains variable, or proposal/current density terms are mismatched. |
| Promotion criterion | Analytic independent-MH fixture, exact theta-measure ratio, paired artifacts, finite/status gates. |
| Promotion veto | Missing `q` term, wrong beta bridge, candidate not sampled from declared `q`, invalid endpoint accepted, seed/cloud mismatch, nonfinite artifact, or overwritten root. |
| Continuation veto | Target/proposal unavailable, exact fixture contradiction, three unrepaired infrastructure failures, or budget exhaustion. |
| Repair trigger | Descriptive support result; no proxy threshold is promoted. |
| Explanatory diagnostics | Acceptance by beta, movement, ESS, roots, mode mass, support ranges, and paired spread. |
| Nonclaims | No finite-run convergence, IID Gaussian law, posterior correctness, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default promotion. |

## Evidence contract

The comparator is the identity arm from the same regenerated initial cloud and
the same deterministic resampling seeds. The primary hard gates are:

1. A CPU-hidden analytic independent-MH fixture verifies beta-zero acceptance,
   finite log ratios, and nonzero movement under a shifted target.
2. The q=20 runner verifies the exact Phase 47 pilot contracts, target
   signature, proposal geometry, and initial tensor hashes.
3. Every independent candidate is sampled from the declared normalized
   defensive mixture, and its stored proposal log density is recomputed by the
   same theta-density function used in the acceptance ratio.
4. Candidate invalid/status rows are rejected and never replace current rows.
5. All endpoint tensors are finite `[256,4]` theta rows; identity and
   independent arms use the same resampling seeds.

Acceptance, ESS, mode mass, roots, and spread comparisons are explanatory only.
Three paired replicates and two mutation steps per nonterminal beta stage are a
diagnostic design, not uncertainty-supported evidence for ranking.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| independent proposal `q` | fixed Phase 28 defensive mixture | stale geometry or wrong density measure | proposal recomputation and protocol hash | reviewed hypothesis |
| defensive epsilon `0.20` | frozen Phase 47/Phase 28 comparator | tail mass changes acceptance/support | record raw candidate components and q values | hypothesis, not default |
| two steps per stage | Phase 47 matched budget | insufficient rejuvenation | per-stage acceptance/movement | bounded diagnostic choice |
| N=256, three replicates | Phase 47 paired design | finite-replicate noise | raw replicate table, no ranking | diagnostic design |
| candidate seed offset `30000` | new non-overlap from resampling `1000` and local MH `20000` | accidental RNG coupling | manifest and per-stage seed ledger | reviewed implementation choice |
| GPU/XLA target lane | repository NeuTra/target default | allocator or compile failure | pre-import growth verification | required execution choice |
| CPU-hidden report | diagnostic policy | mistaken GPU/default claim | explicit manifest | diagnostic only |

No numeric choice is promoted because it appeared in an earlier phase.

## Skeptical pre-execution audit

The plan passes on 2026-08-26. The independent-MH ratio includes both current
and candidate `log q` terms, so it targets the stated bridge rather than an
unstated target. At beta zero the proposal and bridge are identical, giving an
exact algebraic smoke condition. Candidate sampling and density evaluation use
the same defensive mixture and theta measure. The regenerated initial tensor
hash must match the Phase 47 paired boundary, preventing a different starting
cloud from masquerading as a mutation effect. Acceptance is not a promotion
metric, and a finite support reduction cannot establish whitening.

## Pre-mortem

| Misleading outcome | Distinguishing check | Response |
|---|---|---|
| independent arm starts from a different cloud | exact Phase 47 initial hashes and paired hash receipt | hard veto; discard interpretation |
| candidate sampled from one component but scored as mixture | component labels plus exact mixture-log-density recomputation | hard veto; repair candidate generator |
| beta-zero candidates are rejected | analytic fixture and first-stage acceptance receipt | hard veto; inspect ratio algebra |
| high acceptance reflects repeated proposal draws, not target movement | accepted count, component labels, displacement, and beta-wise rows | explanatory only; no ranking |
| support spread decreases because q itself is biased | compare unchanged q support and downstream target checks | retain role-limited result; do not promote whitening |

## Artifacts and commands

Fixture (CPU-only, before q=20):

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase48_fixture_2026_08_26.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase48-independent-proposal-mutation/fixture
```

The q=20 boundary runner requires a visible trusted GPU and
`TF_FORCE_GPU_ALLOW_GROWTH=true`; it writes a fresh `q20-paired/` root under
`phase48-independent-proposal-mutation/`. The report is CPU-hidden and
read-only.

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `independent_mh_reduces_between_bank_variability_descriptive` | nonlocal invariant mutation is descriptively favorable under paired finite diagnostics | retain as role-limited candidate; plan longer validation with uncertainty and downstream target checks |
| `independent_mh_does_not_reduce_variability` | nonlocal mutation does not repair the observed finite support variability in this scope | investigate proposal/objective support separately; keep whitening/HMC/LEDH closed |
| `independent_mh_hard_veto` | ratio, sampling, measure, or data-boundary failure | repair harness; no scientific interpretation |

No branch is a statistical ranking, posterior result, or whitening result.

Execution completed on 2026-08-26. The passing fixture, GPU boundary, and
CPU-hidden report are recorded under
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase48-independent-proposal-mutation/`.
The report branch was `independent_mh_does_not_reduce_variability`; the repair
refresh and next active subplan are the Phase 48 result/repair notes and
`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase49-subplan-2026-08-26.md`.
