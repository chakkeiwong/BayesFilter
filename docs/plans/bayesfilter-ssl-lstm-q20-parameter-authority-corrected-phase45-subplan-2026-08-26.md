# Corrected Parameter-Authority Phase 45 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.7-independent-n512-replication`  
Entry gate: Phase 44 passed its hard boundary/report and retained the whitening veto  
Status: `PASS_V2_7_INDEPENDENT_N512_REPORT_REPAIR_TRIGGERED`  
Local cap: 5400 s

## Harness repair ledger

An initial mechanical copy of the Phase 44 scripts was rejected by the
pre-execution compile gate: it retained four-bank/v2.6 assumptions and had a
syntax error. No Phase 45 experiment ran from that copy and no scientific
artifact was produced. The scripts were replaced with purpose-built v2.7
runner/reporter implementations, then passed Python compilation,
`git diff --check`, and a CPU-hidden reporter import smoke. The runner's
CPU-hidden refusal is intentional; the trusted audit must launch with a
visible GPU and `TF_FORCE_GPU_ALLOW_GROWTH=true`.

The fresh N=512-b pilot was generated only after that repair. It passed
`PASS_THETA_MEASURE_PILOT`; its receipt and tensor hashes are distinct from
authority, A, B, C, and N512-a. Two earlier inspection snippets used stale
diagnostic field names and failed read-only; those failures did not modify or
invalidate the pilot.

## Question

Does a second independently generated `N=512` theta bank reproduce the first
N=512 support/residual behavior under one unchanged frozen NeuTra trainer per
arm, or does support variability persist even at the larger bank size?

This is a finite-support replication diagnostic. It is not an objective,
architecture, whitening, posterior, HMC, or canonical LEDH experiment.

## Frozen boundary and changed scope

The target remains the batch-native q=20 SSL-LSTM target with `theta in R^4`.
The 60D UKF state is internal and is never a particle coordinate. The old v2.2
root-group training rows (232 rows), normalized weights, proposal semantics,
M0/C0 protocol hashes, four arm configurations, 200 optimizer updates,
target signature, GPU/XLA settings, and Phase 44 terminal state hashes are
frozen. The first N=512 bank, plus N=256 banks A, B, and C, remain untouched
contextual audits. The only new scientific input is a fresh N=512 pilot bank
with seed `(20260826, 4505)` and calibration size 128.

Each arm constructs one trainer, trains only on the old training rows, and
evaluates A, B, C, N512-a, and N512-b only after the final update. The
authority cloud is retained as support context but is not called independent:
its root-group subset supplies the optimizer measure. None of the five audit
banks is used for training, tuning, checkpoint selection, or objective
selection. The calibration cloud is diagnostic-only and is not part of the
claim bank or optimizer measure.

## Research-intent ledger

| Field | Phase-45 statement |
|---|---|
| Main question | Does independent N=512 replication reduce uncertainty about finite-bank support variability? |
| Candidate mechanism | A second independent N=512 post-training theta bank under a frozen trainer. |
| Comparator | First N=512 bank, N=256 A/B/C, authority support context, and the historical v2.2 validation row. |
| Expected failure | The two N=512 banks differ materially or both retain non-Gaussian residuals. |
| Promotion criterion | Only engineering, measure, status, independence, and state-identity gates. |
| Promotion veto | Any copied/stale bank, protocol or target mismatch, fresh-row use, non-finite/status-invalid target, shape mismatch, GPU-policy failure, or state-hash mismatch. |
| Continuation veto | Target/common support unavailable, repeated unrepaired infrastructure failure, platform block, or exhausted campaign budget. |
| Repair trigger | A hard harness failure or a descriptive branch requiring a separately scoped support/proposal/objective experiment. |
| Explanatory diagnostics | ESS, root count, mode mass, target/proposal ranges, log-ratio ranges, transport moments, and bank-to-bank residual differences. |
| Nonclaims | No IID Gaussian theorem, exhaustive mode discovery, posterior theorem, normalizer, HMC readiness, canonical LEDH validity, superiority, or default promotion. |

## Evidence contract

**Hard gates before interpretation:**

1. The new pilot is `PASS_THETA_MEASURE_PILOT`, finite, `theta_R4`, target
   signature exact, and has the frozen M0/C0 protocol hashes.
2. Authority, A, B, C, N512-a, and N512-b pilot receipts are distinct; all
   required M0/C0 tensor hashes are distinct.
3. The frozen root-group split is complete, row-disjoint, and root-disjoint.
4. One batch-native GPU/XLA trainer per arm consumes only old training rows;
   the five independent audit banks are observed only after the final update.
5. Every terminal trainer state hash equals its Phase 44 reference hash.
6. All target/status, score, transport-parity, and support tensors are finite.

Residual and support differences are explanatory diagnostics. They cannot
promote whitening, posterior correctness, HMC, or an objective change.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| N=512 claim bank | Phase 44 repair branch | particle count confounded with protocol drift | count, protocol, and target receipt checks | reviewed diagnostic hypothesis |
| second seed `(20260826,4505)` | fresh campaign allocation | accidental correlation or copied bank | receipt/tensor hash separation | reviewed campaign choice |
| calibration size 128 | fixed from Phase 44 replication scope | calibration behavior mistaken for claim evidence | separate calibration receipt and explicit non-use fields | unpromoted hypothesis |
| 200 updates | Phase 44 frozen-state comparison | undertraining mistaken for support effect | exact state-hash equality | frozen comparator |
| all prior banks retained | Phase 44 context | outcome-dependent bank dropping | five audit banks plus authority support context and no selection flags | mandatory context |
| old validation comparator | v2.2 historical artifact | tiny holdout is not population truth | descriptive-only label and red-team note | explanatory only |

## Skeptical pre-execution audit

The plan passes review on 2026-08-26. It changes only one independent
post-training bank draw. It leaves the target, proposal semantics, optimizer
measure, arm settings, terminal-state identity, hardware class, promotion
vetoes, and whitening criterion unchanged. The report contains both N=512
banks and all prior context, so no outcome-dependent bank is removed. A passing
command answers the stated replication question because every bank is evaluated
under a checked common trainer state.

## Pre-mortem

| Misleading failure | Distinguishing check | Response |
|---|---|---|
| New bank silently uses a different proposal | exact target/protocol/measure hashes | hard veto before training |
| Calibration or fresh rows leak into training | pilot receipts and explicit false-use fields | hard veto; preserve root |
| Trainer reconstruction changes | exact Phase 44 state hashes | classify as harness/determinism repair |
| Two N=512 banks appear similar only under a proxy threshold | use exact order relations and retain raw moments | descriptive branch only |
| Both N=512 banks share proposal-support bias | target/proposal ranges, root/mode table, red-team note | do not claim coverage; scope proposal repair |

## Artifacts and commands

Fresh pilot output:

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/fresh-n512-b/`

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/fresh-n512-b \
  --particles 512 --calibration-particles 128 --arms both --seed 20260826 4505
```

The trusted audit consumes the Phase 44 authority/A/B/C/N512-a receipts and
the new N512-b receipt:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase45_2026_08_26.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --fresh-root-a docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-a-n256 \
  --fresh-root-b docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-b-n256 \
  --fresh-root-c docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase43-third-bank-support/fresh-c-n256 \
  --fresh-root-n512-a docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase44-larger-n-support/fresh-n512 \
  --fresh-root-n512-b docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/fresh-n512-b \
  --reference-audit-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase44-larger-n-support/frozen-four-bank-audit-attempt2 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/frozen-five-bank-audit \
  --steps 200 --seed 20260826 4211
```

The trainer seed `20260826,4211` is frozen from Phase 44 so the state-hash
comparison remains meaningful. The new bank's independent pilot seed is
`20260826,4505`; these are separate seed roles.

The CPU-hidden report consumes that audit and writes
`phase45-independent-n512-replication/report/` with a unique output root.

Trusted GPU audit and CPU report use the Phase 45 scripts and unique roots
under `phase45-independent-n512-replication/`. A failed attempt receives a
new suffix; prior artifacts are never overwritten. The run manifest records
the command, git commit/dirty state, environment, seeds, GPU memory policy,
source hashes, wall time, and artifact paths.

## Interpretation branches

| Observed branch | Interpretation role | Next action |
|---|---|---|
| Both N=512 banks have lower mean and covariance residuals than A in every arm, and both are also below the old comparator in every arm | the exact A-outlier order relation replicated | retain support hypothesis; write a scoped proposal/support adjudication |
| Both N=512 banks have lower mean and covariance residuals than A in every arm, but at least one is not below the old comparator everywhere | the A order replicated but finite support remains mixed | repair/adjudicate proposal support before objective changes |
| Either N=512 bank fails the lower-than-A order in any arm | the Phase 44 A ordering did not replicate | support variability persists at N=512; repair proposal/support generation |
| Any hard gate fails | engineering/numerical veto | preserve failed root, repair, rerun fresh |

No branch is a statistical ranking. A passing replication does not establish a
population limit or IID Gaussian law.

## Closure and inter-phase repair

The pilot, five-bank GPU audit, and CPU report completed on 2026-08-26. The
audit status was `PASS_V2_7_INDEPENDENT_N512_BOUNDARY`; its wall time was
`1617.6877457919763 s`. All four terminal state hashes matched the Phase 44
reference, all finite/status/transport and GPU memory-growth/XLA/TF32 gates
passed, and no fresh bank was used for training or selection. The report
status was `PASS_V2_7_INDEPENDENT_N512_REPORT` with branch
`n512_replication_order_reproduced_but_support_mixed`.

The branch means that both N=512 banks reproduce the lower-than-A ordering in
all four arms, while the two-bank result is not uniformly below the historical
old comparator. This strengthens a finite-support explanation but does not
establish common support, whitening, a population limit, or an objective
defect. The whitening and HMC/LEDH vetoes remain active.

Artifacts:

- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/fresh-n512-b/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/frozen-five-bank-audit/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/report/result.json`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase45-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase45-repair-refresh-2026-08-26.md`

The refreshed next version is `v2.8-support-envelope-diagnostic`. Phase 46
generates one untouched N=512-c bank and computes a descriptive proposal/
support envelope across authority, A, B, C, N512-a, N512-b, and N512-c. It
does not retrain or alter the frozen NeuTra state.
