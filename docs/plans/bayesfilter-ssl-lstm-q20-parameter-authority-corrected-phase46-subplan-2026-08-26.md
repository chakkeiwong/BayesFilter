# Corrected Parameter-Authority Phase 46 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.8-support-envelope-diagnostic`  
Entry gate: Phase 45 report completed with branch `n512_replication_order_reproduced_but_support_mixed`  
Status: `IN_PROGRESS`  
Local cap: 2400 s

## Question and scope

Does a third independently generated `N=512` theta bank fall inside the
support/proposal envelope of the first two N=512 banks, or does support
variability persist? The diagnostic uses the fixed theta proposal law and the
stored target/proposal receipts from all banks. It does not retrain NeuTra,
alter the target, change the proposal schedule, or evaluate HMC/LEDH.

The declared target remains `theta in R^4`; the 60D UKF state is internal.
Authority is the frozen training measure, not an independent audit bank.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Is the N=512 support pattern stable across three independent draws? |
| Mechanism | Fresh N512-c pilot plus fixed-law support-envelope calculation. |
| Comparator | N512-a/N512-b envelope, then A/B/C and authority context. |
| Expected failure | N512-c leaves the two-bank envelope or proposal-density support remains separated. |
| Promotion criterion | Only pilot, hash, measure, finite-value, and recomputation gates. |
| Promotion veto | Any stale/copy bank, density mismatch, nonfinite tensor, wrong dimension/protocol, or data-use violation. |
| Continuation veto | Target/proposal unavailable, three unrepaired infrastructure failures, or budget exhaustion. |
| Repair trigger | Envelope instability or persistent support separation. |
| Explanatory diagnostics | ESS, mode mass, roots, log-density ranges, coordinate-box inclusion, and raw pairwise distances. |
| Nonclaims | No whitening, population support theorem, mode-discovery theorem, posterior, superiority, HMC, or LEDH claim. |

## Evidence contract

The exact baseline is the fixed M0 theta proposal law already used in Phases
27--45. The primary hard screen is receipt integrity: the new bank and every
retained bank must be finite, dimension-four, target/status valid, and carry
the exact target/protocol signatures. The support envelope is explanatory only.

For each bank, the report records raw weighted summaries and evaluates the
fixed proposal log density at that bank's theta rows. For every ordered pair of
banks `(i,j)`, it reports the weighted fraction of bank `i` rows inside the
coordinate-wise min/max box of bank `j`, with no threshold or promotion claim.
It also records the exact coordinate-box intersection/union and pairwise
Euclidean nearest-neighbor summaries. These are finite empirical diagnostics,
not density or coverage proofs.

The artifact is a unique report root under
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase46-support-envelope/`.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early check | Status |
|---|---|---|---|---|
| N512-c | fresh campaign seed `(20260826,4606)` | copied/correlated bank | pilot and tensor hashes | reviewed diagnostic hypothesis |
| M0 proposal law | frozen Phase 28 protocol | stale geometry or epsilon | recompute stored proposal logs | frozen comparator |
| coordinate min/max box | exact finite descriptive statistic | box overstates high-dimensional support | retain raw pairwise values and no threshold | explanatory only |
| Euclidean nearest-neighbor distance | direct theta-space diagnostic | scale dependence | report raw units and do not rank methods | explanatory only |
| CPU hidden report | policy-approved diagnostic lane | cannot support GPU/default claims | manifest records hidden GPU | diagnostic only |

## Skeptical pre-execution audit

The plan passes on 2026-08-26. It does not use transport residuals as a
promotion criterion, does not pool or drop any bank, and does not introduce a
new objective. The proposal density is recomputed from the same geometry and
protocol rather than trusted from labels. The exact box and distance metrics
answer the finite support question while preserving their proxy status.

## Pre-mortem

| Misleading result | Distinguishing check | Response |
|---|---|---|
| N512-c appears stable because the copied pilot is identical | pilot and tensor SHA-256 checks | hard veto; preserve root |
| stored proposal density is stale | recomputation max residual and protocol hash | hard veto; no interpretation |
| coordinate boxes suggest coverage in four dimensions | report pairwise raw values and no threshold | keep support claim closed |
| target log ranges agree but modes differ | roots, sign mass, and theta summaries | classify as support variability |
| CPU report is mistaken for GPU evidence | explicit hidden-GPU manifest | diagnostic-only label |

## Artifacts and commands

Pilot (CPU diagnostic lane):

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase46-support-envelope/fresh-n512-c \
  --particles 512 --calibration-particles 128 --arms both --seed 20260826 4606
```

The read-only report consumes the authority, A, B, C, N512-a, N512-b, and
N512-c pilot roots and writes `phase46-support-envelope/report/`. It is
CPU-only, uses no trainer, and refuses to overwrite an existing root.

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `n512_c_inside_two_bank_scalar_envelope` | scalar support summaries for N512-c lie between N512-a/b on every predeclared field | retain support hypothesis; consider a separately planned downstream objective test only after fresh validation |
| `n512_c_outside_two_bank_scalar_envelope` | support variability remains at N512 | keep objective/whitening closed; repair proposal/support generation |
| hard gate failure | engineering/data-integrity veto | preserve root and repair with a fresh root |

Neither descriptive branch is a statistical ranking or a whitening result.
