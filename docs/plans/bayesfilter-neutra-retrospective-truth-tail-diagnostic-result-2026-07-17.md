# NeuTra Retrospective Truth-Tail Diagnostic Result

Date: 2026-07-17

Decision: `RETROSPECTIVE_DIAGNOSTIC_COMPLETE_NO_SECOND_SEED_NEEDED`

## Outcome

The retrospective CPU-only diagnostic completed for all four eligible
learned-NeuTra configurations. Preserved source-result hashes, retained-sample
hashes and shapes, target signatures, modern R-hat, bulk/tail ESS, final health,
target status, divergence telemetry, and warm-up separation all passed before
posterior tails were interpreted.

| Cell | Parameters | Central truth | Minimum `p_truth` | Classification |
| --- | ---: | --- | ---: | --- |
| `LGSSM-EXACT` | 18 | yes | `0.0678083` (`q2`) | `ONE_SEED_DIAGNOSTIC_PASS` |
| `PP-UKF` | 6 | no | `0.212549` (`K`) | `RETROSPECTIVE_ONE_SEED_TAIL_PASS_NONCENTRAL_TRUTH` |
| `PP-SGQF` | 6 | no | `0.215049` (`K`) | `RETROSPECTIVE_ONE_SEED_TAIL_PASS_NONCENTRAL_TRUTH` |
| `SIR-SGQF` | 3 | yes | `0.372664` (`nu`) | `ONE_SEED_DIAGNOSTIC_PASS` |

All 33 generating values are inside their empirical 95% posterior intervals.
No parameter has `p_truth < 0.05`; no parameter is marginal or severe; no fresh
dataset seed is nominated by the predeclared ladder.

The direct scientific conclusion is:

> Learned NeuTra passed a one-seed central-truth diagnostic for the tested exact
> LGSSM and parameterized SIR-SGQF configurations. The tested predator-prey UKF
> and SGQF configurations also passed the numerical one-seed truth-tail screen,
> but their frozen fixture was not fully generated at the prior mean.

## Claimed And Computed Quantities

| Item | Verdict |
| --- | --- |
| Claimed target | parameter-wise location of the known generating truth in each preserved filter-defined NeuTra posterior |
| Quantity computed | smoothed two-sided posterior truth-tail probability `2 min(F_truth, 1-F_truth)` from 16,000 retained draws per parameter |
| Equality to requested diagnostic | correct: this is the predeclared posterior-tail diagnostic, using half-weight ties and `0.5/(N+1)` boundary smoothing |
| 95% interval relation | explanatory cross-check; interval inclusion did not replace `p_truth` as the decision criterion |
| Sampler evidence | reused and hash-verified from the original claim-bearing HMC runs; it was not recomputed by the reporting pass |
| Unproved | repeated-seed coverage, calibration, exactness of UKF/SGQF relative to the scientific model, and broader reliability |

All displayed summaries use model coordinates: physical LGSSM coefficients and
standard deviations, physical predator-prey parameters, and physical SIR scale
parameters. Each chart is strictly increasing parameter-wise, so the truth-tail
probability equals the value in the source/raw chart, while the reported means
and intervals remain interpretable in model coordinates.

## Eligibility And Provenance

LGSSM truth is the raw-coordinate prior center specified by the frozen source
contract and maps to the declared physical transition and standard-deviation
template. SIR uses generating log scales `(0,0,0)`, equal to the declared Normal
prior means.

Predator-prey uses independent physical uniforms. Four generating parameters
equal their prior means, but `K=114` versus prior mean `120` and `s=0.3` versus
prior mean `0.6`. The two predator-prey cells are therefore not eligible for the
literal prospective central-truth label. Their qualified classification is a
visible correction, not a failure.

## Sampler Veto Screen

| Diagnostic | LGSSM | PP-UKF | PP-SGQF | SIR-SGQF |
| --- | ---: | ---: | ---: | ---: |
| maximum modern R-hat | `1.0021492` | `1.0008111` | `1.0003276` | `1.0000689` |
| minimum bulk ESS | `4,571.61` | `27,623.60` | `26,978.49` | `16,358.48` |
| minimum tail ESS | `3,976.95` | `13,394.13` | `12,974.65` | `14,568.53` |
| warm-up / retained per chain | `2,000 / 4,000` | `2,000 / 4,000` | `2,000 / 4,000` | `2,000 / 4,000` |
| hard vetoes | none | none | none | none |
| target/status/finite health | pass | pass | pass | pass |
| energy-error divergence count | `0` | `0` | `0` | `0` |

Modern R-hat means the maximum of rank-normalized split and folded
rank-normalized split R-hat. TFP HamiltonianMonteCarlo did not expose a native
divergence flag in these preserved runs. The recorded divergence diagnostic is
the predeclared nonfinite-log-acceptance or `log_accept_ratio < -1000` screen;
the result does not claim otherwise.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | retrospective diagnostic complete; no second seed |
| Primary criterion | passed for all 33 parameters: `p_truth >= 0.05` |
| Veto diagnostic status | clear for all four configurations |
| Main uncertainty | one favorable synthetic dataset per configuration; two predator-prey truth coordinates are not prior-centered |
| Next justified action | use the same prospective one-seed design on the next runnable model/filter cell; do not rerun these four solely for this screen |
| Not concluded | calibration, coverage, universal reliability, filter exactness/ranking, sampler superiority, production readiness, or default readiness |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | no hard veto supported for the four preserved runs |
| Statistically supported ranking | none; no methods were ranked |
| Descriptive-only differences | cell-to-cell `p_truth`, means, SDs, intervals, ESS, acceptance, and runtime |
| Default readiness | not established |
| Next evidence needed | a new model/filter configuration, or one fresh seed only if a future first seed is marginal |

## Run Manifest And Artifacts

Exact command:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tf-gpu \
  python docs/benchmarks/compute_neutra_retrospective_truth_tail.py \
  --output-root docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01
```

- reporting wall time: `4.5677` seconds;
- environment: `tf-gpu`, Python `3.11.14`, TensorFlow `2.19.1`, TFP `0.25.0`;
- GPU status: deliberately hidden with `CUDA_VISIBLE_DEVICES=-1`; TensorFlow
  reported no visible physical GPU;
- structured result:
  `docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01/result.json`,
  SHA-256 `3d7be7f1079b1947888b7e4cb0e26706df6c38a09ac42419c1d4423613dd9c6d`;
- human-readable parameter table:
  `docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01/result.md`,
  SHA-256 `2d42d3cfa5c9998b9121e96d9a1120f012584f92b196c00a7ea14ce1c9fb50da`;
- run manifest:
  `docs/plans/artifacts/neutra-retrospective-truth-tail-20260717/attempt-01/run_manifest.json`,
  SHA-256 `306e03d9eec21c954f7e6cea4c9301cb5833efa7d1cb9f8c7b14f031499b9356`.

TensorFlow emitted CUDA plugin registration and `cuInit` messages despite the
explicit CPU hide, plus TFP complex-to-real ESS implementation warnings. These
did not make a GPU visible, change the deterministic results, or trip any
finite/ESS check. They are execution noise, not evidence that GPU work occurred.

## Post-Run Red Team And Drift Audit

The strongest alternative explanation is favorable fixture choice: truth was
at the prior center for LGSSM/SIR and partly prior-centered for predator-prey.
This screen asks only whether truth is in a non-extreme posterior location for
one dataset; it cannot establish repeated-sampling coverage.

The weakest pass is LGSSM `q2` at `p_truth=0.0678083`, above but closest to the
`0.05` threshold. The threshold was frozen before computation. Lowering it after
seeing the result or rerunning merely because it is close would violate the
cost-bounded ladder. A fresh independent LGSSM seed with the same parameter
repeatedly below `0.05` would weaken the one-fixture conclusion, but no second
seed is triggered by the current policy.

Post-run audit verdict: `NO_MATERIAL_DRIFT`.

- The baseline remained generating truth, not plain-HMC means.
- The primary criterion remained `p_truth`; intervals and means stayed
  explanatory.
- No source result, sample archive, target, training, tuning, or HMC run was
  modified or regenerated.
- All four source-result and retained-archive hashes matched frozen expected
  values before analysis.
- Warm-up remained separate and was not pooled into posterior summaries.
- No Python sample-axis loop or NumPy implementation was added; TensorFlow/TFP
  vectorized the reporting computations.
- No GPU run, second seed, new training, target evaluation, or scientific claim
  beyond the evidence contract occurred.
- Claude was not called because the parent program had already reached its
  review ceiling; the required skeptical plan audit and terminal Codex audit
  were completed locally.

Nothing in this result establishes universal NeuTra reliability. It adds two
literal central-truth one-seed passes and two qualified retrospective tail
passes to the existing nine-family, twelve-configuration experience ledger.
