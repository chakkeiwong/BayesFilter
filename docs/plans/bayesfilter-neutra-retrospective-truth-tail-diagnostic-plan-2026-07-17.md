# NeuTra Retrospective Truth-Tail Diagnostic Plan

Date: 2026-07-17

Status: `EXECUTED_COMPLETE`

Execution result:
`docs/plans/bayesfilter-neutra-retrospective-truth-tail-diagnostic-result-2026-07-17.md`

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | Do the preserved learned-NeuTra retained draws place the known generating parameters in non-extreme posterior locations for LGSSM, PP-UKF, PP-SGQF, and SIR-SGQF? |
| Candidate under test | The already frozen learned-NeuTra HMC result for each named model/filter configuration |
| Expected failure mode | A generating parameter lies in a marginal or severe posterior tail despite otherwise valid sampler diagnostics |
| Promotion criterion | Valid preserved sampler evidence and every parameter has smoothed two-sided `p_truth >= 0.05` |
| Promotion veto | Any parameter has `p_truth < 0.05` |
| Continuation veto | Artifact/provenance mismatch, invalid sampler evidence, missing truth, or missing retained draws; do not interpret posterior tails |
| Repair trigger | A marginal `0.003 <= p_truth < 0.05` result nominates one fresh data seed under a separate serious-run continuation plan |
| Explanatory diagnostics | Posterior mean, SD, 95% empirical interval, mean ESS, ECDF counts, and distance from the prior center |
| Forbidden conclusion | Calibration, coverage, universal reliability, filter exactness, sampler superiority, production readiness, or default readiness |

## Scope And Evidence Contract

The diagnostic is retrospective and reporting-only. It does not rerun a target,
transport training, tuning, warm-up, or HMC. The exact baseline is the known
generating truth for the same preserved dataset and posterior target. Plain HMC
is not a promotion comparator.

For parameter `j`, pool the four retained chains only after verifying their
preserved sampler validity. With `N` pooled draws, define

\[
F_j(\theta_j^{\mathrm{true}}) =
\frac{n_{<} + 0.5 n_{=} + 0.5}{N+1}, \qquad
p_{\mathrm{truth},j}=2\min(F_j,1-F_j).
\]

The half-weight for ties and the `0.5/(N+1)` boundary smoothing avoid exact
zero or one. The diagnostic uses the model coordinates in which results are
scientifically interpretable:

- LGSSM: transition coefficients, process standard deviations, and observation
  standard deviations;
- predator-prey: the six physical parameters;
- SIR-SGQF: physical `kappa`, `nu`, and observation standard deviation scale.

All transforms are strictly increasing parameter-wise, so `p_truth` is
unchanged by computing in raw/source or displayed physical coordinates.

### Decision Ladder

| Condition | Classification |
| --- | --- |
| sampler valid, fully central-truth fixture, every `p_truth >= 0.05` | `ONE_SEED_DIAGNOSTIC_PASS` |
| sampler valid, not fully central-truth, every `p_truth >= 0.05` | `RETROSPECTIVE_ONE_SEED_TAIL_PASS_NONCENTRAL_TRUTH` |
| none below `0.003`, at least one below `0.05` | `MARGINAL_ONE_SEED`; nominate exactly one fresh dataset seed |
| any below `0.003` | `ONE_SEED_DIAGNOSTIC_FAILURE`; stop and investigate |
| sampler/provenance/finite checks fail | `SAMPLER_INCONCLUSIVE`; stop without tail interpretation |

The central-truth condition is evaluated in each target's declared prior
coordinate system. LGSSM truth equals the independent Gaussian raw-coordinate
prior center. SIR truth equals the independent Normal log-scale prior center.
Predator-prey uses independent physical uniforms; `K=114` versus prior mean
`120` and `s=0.3` versus prior mean `0.6`, so its two filter cells cannot receive
the prospective central-truth label even if their tail checks pass.

### Sampler Validity Veto

The claim-bearing source result must record all of the following:

- completed and passed retained run;
- modern R-hat `max(rank-normalized split, folded rank-normalized split) <= 1.01`;
- bulk ESS `>= 1000` and tail ESS `>= 400` for every parameter;
- no hard veto;
- finite retained samples, finite target log probabilities, valid target status,
  and passed health record;
- zero divergences under the preserved energy-error definition; and
- warm-up separately retained and excluded from posterior summaries.

TFP HMC did not expose a native divergence flag in these runs. The result must
say this explicitly; the preserved nonfinite/`log_accept_ratio < -1000` energy
screen is the available divergence diagnostic. This limitation is reported and
does not get silently represented as a native divergence check.

### Required Artifacts

- `docs/benchmarks/compute_neutra_retrospective_truth_tail.py`
- `docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01/result.json`
- `docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01/result.md`
- `docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01/run_manifest.json`
- `docs/plans/bayesfilter-neutra-retrospective-truth-tail-diagnostic-result-2026-07-17.md`

The JSON result binds the source result and sample archive SHA-256 hashes,
truth/prior provenance, shapes, target signatures, parameter-wise statistics,
and classifications.

## Default And Assumption Audit

| Choice | Provenance and justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| thresholds `0.05` and `0.003` | owner directive in the active NeuTra evidence ledger | arbitrary thresholds mistaken for a theorem | print thresholds and nonclaims in every result | reviewed policy |
| one data seed first | owner cost directive | fixture luck mistaken for calibration | label one-seed scope and parameter count | reviewed policy |
| smoothed ECDF | prospective ledger requires finite-sample smoothing; formula frozen above | implementation convention changes a boundary result | unit-check hand-constructed arrays | reviewed default |
| 95% interval | user requested credible-interval context | interval inclusion replaces tail criterion | make `p_truth` primary and interval explanatory | explanatory only |
| four-chain pooled ECDF | all four configurations have valid 4-chain retained archives | invalid chains contaminate the ECDF | preserved all-chain convergence and health veto first | reviewed default |
| mean ESS | matches existing confirmation scripts and reports MC precision for the mean | confused with rank bulk/tail ESS | report all three ESS roles separately | explanatory only |
| current target source files | historical target contracts bind source hashes, while the diagnostic reads preserved samples | later source edits rewrite history | use constants only as corroboration; bind source result and archive hashes | convenience, bounded |

## Skeptical Plan Audit

Audit verdict: `PASS_AFTER_REVISION`.

The initial idea incorrectly risked applying the prospective central-truth label
to all four configurations. Inspection showed that predator-prey truth is not
at the physical-uniform prior mean for two of six parameters. The plan now
separates the numerical posterior-tail screen from the central-truth claim and
uses a qualified retrospective label for PP-UKF and PP-SGQF.

The audit also checked:

- wrong baseline: generating truth, not plain-HMC means, is the comparator;
- proxy promotion: posterior means, SDs, intervals, acceptance, and runtime are
  explanatory only;
- stop conditions: provenance or sampler invalidity stops interpretation;
- stale context: active P4 attempt-02/03, P6 attempt-01, and LGSSM repaired
  dense-seed1201 artifacts are selected, not superseded attempts;
- environment: CPU is deliberately hidden with `CUDA_VISIBLE_DEVICES=-1` and
  no target or GPU computation occurs;
- answerability: retained tensors contain all draws needed for ECDFs, and
  source results contain modern R-hat, ESS, health, target, and archive bindings;
- fairness: no method ranking is attempted; and
- hidden assumption: monotone coordinate transforms preserve tail ranks but
  not means/intervals, so summaries are emitted in displayed model coordinates.

Claude is not called: the parent multi-model program already reached its
declared Claude review ceiling. The active review policy makes reviewer
unavailability non-blocking, and this plan receives a fresh skeptical Codex
audit instead.

## Execution

Exact command:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tf-gpu \
  python docs/benchmarks/compute_neutra_retrospective_truth_tail.py \
  --output-root docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01
```

Budget: one CPU reporting attempt, expected under five minutes, no GPU, no new
data, no training, and no HMC. A localized reporting bug may be repaired once
in `attempt-02`; numerical or provenance failures stop.

After execution:

1. run the script's unit self-check and artifact verification;
2. inspect all parameter rows and classification logic;
3. write the close result and run manifest;
4. run `git diff --check` on the new files; and
5. perform a post-run drift audit against this evidence contract.

No fresh second-seed run is launched unless a marginal classification occurs.
If it occurs, write a separate target-specific serious-run continuation plan
with a GPU budget before training or HMC. A severe failure stops immediately.

## Execution Closure

Attempt 01 completed in 4.57 seconds of reporting time with deliberately hidden
GPU devices. Both central-truth configurations passed, both predator-prey cells
passed the same numerical tail screen with the required noncentral-truth
qualifier, and no parameter was marginal or severe. No second seed was
nominated. The exact classifications, parameter rows, hashes, decision tables,
and post-run drift audit are in the execution result linked above.
