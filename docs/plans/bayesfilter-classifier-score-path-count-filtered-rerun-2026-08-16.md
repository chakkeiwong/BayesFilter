# Filtered SIR Path-Count Rerun

Date: 2026-08-16
Status: `EXECUTING`

## Research Intent

Rerun the SIR independent classifier-score ladder at `N=16,384` and
`N=32,768` using the explicitly requested pairwise invalid-path removal arm.
The existing `N=8,192` original-law campaign remains historical context only;
these new stages are survivor-conditioned training results.

## Frozen Execution

- Model: SIR observation simulator, `T=(20,40,50)`, `j=(0,1,2)`.
- Paths: per class and perturbation, with six declared perturbations.
- Policy: `remove_invalid_paths`; remove `+/-` rows as matched pairs.
- Acceptance rule: accept a bundle when removed pairs divided by `6*N` are
  below `0.001` (0.1%); report raw invalid-row rate separately.
- Hardware: repository GPU default, preferred available physical GPU 1,
  fallback GPU 0; `tftwogpu`, TensorFlow/XLA, memory growth.
- Fresh roots: `sir_16384_filtered_campaign_attempt02` and
  `sir_32768_filtered_campaign_attempt01` under the V7 artifact root.
- No result directory is overwritten; the campaign is resumable per bundle.

## Skeptical Audit

The removal step changes the training law, so it cannot support the original
SIR likelihood-ratio claim. It is retained only because the user explicitly
requested this sensitivity run. The threshold is an acceptance/reporting rule,
not evidence that censoring is scientifically innocuous. A single stage is not
enough for a valid variance or `1/N` claim unless all ten bundles complete and
the results are aggregated with the predeclared paired analysis.

## Stop Conditions

Stop a stage on invalid artifact, source/hash mismatch, non-finite fit,
optimizer incompletion, GPU/memory failure, or exhausted compute budget.
After both stages, report per-bundle generated rows, invalid rows, removed
pairs, acceptance status, score outputs, and any aggregate variance comparison.
