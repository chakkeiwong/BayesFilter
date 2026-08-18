# NeuTra Control Repair Campaign Result (2026-08-15)

## Outcome

The reviewed repair campaign completed in `518.62 s` (8.64 minutes), below
the `3600 s` cap. It executed 42 fixed-budget GPU/XLA cells with float64,
batch size 4,096, 3,000 reverse-KL updates per cell, TF32 disabled, and
TensorFlow memory growth verified before logical-device initialization. The
campaign and run manifests have valid SHA-256 hashes.

The two proposed repairs were viable on these controls:

- Gaussian LR nomination selected cold joint `LR=1e-3`; both untouched
  confirmation seeds passed every exact-law screen.
- Banana nomination selected the root-preserving permutation with width
  `(32,32)`, initialization scale `0.02`, and `LR=5e-4`; both screening seeds
  and both untouched confirmation seeds passed every exact-law screen.

This does not establish a universal default. It establishes target-specific
viability for the declared Gaussian and banana controls under this exact
transport, budget, and seed protocol.

## Evidence Contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-control-repair-plan-2026-08-15.md` |
| Artifact root | `docs/plans/artifacts/neutra-control-repair-2026-08-15/` |
| Baseline | `(32,32)`, full-reverse permutation, initialization scale `0.02` |
| Gaussian arms | Cold baseline at `2e-4`, `5e-4`, `1e-3`; selection seeds `2,3`; confirmation seeds `4,5` |
| Banana arms | Baseline, identity-biased scale `0.005`, root-preserving permutation, width `(64,64)`; selection seeds `0,1`, screening seeds `2,3`, confirmation seeds `4,5` |
| Selection gate | Both selection seeds pass all exact-law mean, second-moment, and adjacent-cross-moment screens; otherwise minimize maximum standardized discrepancy |
| Confirmation gate | Both untouched confirmation seeds pass all exact-law screens on 131,072 draws |
| Hard vetoes | Nonfinite state/output, invalid GPU/XLA/memory-growth provenance, unequal budget, partition reuse, or exact-law failure |
| Explanatory only | Reverse-KL loss, ESS, ratio SD, max standardized discrepancy, clipping, gradient norms, runtime |
| Nonclaims | No HMC, SSL-LSTM transfer, multimodal coverage, statistical superiority, universal architecture, or default-readiness claim |
| Git commit recorded | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |

## Results

### Gaussian learning-rate repair

| LR | Selection both seeds | Mean max standardized discrepancy | Mean selection loss |
|---:|---:|---:|---:|
| `2e-4` | No | `5.766` | `5.91269` |
| `5e-4` | Yes | `2.696` | `5.90830` |
| `1e-3` | Yes | `2.631` | `5.90751` |

The repaired nomination selected `1e-3`. Confirmation seed 4 had maximum
standardized discrepancy `2.887`; seed 5 had `2.565`; both passed all three
screen families. This directly repairs the prior failure where a broad
loss-tolerance uncertainty set selected `2e-4`, which failed adjacent
cross-moment screens. The result supports a Gaussian target-specific LR
nomination rule, not a globally optimal LR claim.

### Banana target-specific factors

| Arm | Selected LR | Selection both seeds | Screening max discrepancy (seeds 2/3) | Confirmation |
|---|---:|---:|---:|---:|
| Baseline full-reverse `(32,32)`, scale `0.02` | `5e-4` | No | `3.519`, `4.966` | Not nominated |
| Identity-biased full-reverse `(32,32)`, scale `0.005` | `1e-3` | No | `35.328`, `37.861` | Not nominated |
| Root-preserving `(32,32)`, scale `0.02` | `5e-4` | Yes | `2.548`, `2.845` | 2/2 passed |
| Full-reverse width `(64,64)`, scale `0.02` | `5e-4` | No | `35.383`, `4.193` | Not nominated |

The root-preserving arm passed both confirmation seeds, with maximum
standardized discrepancies `2.863` and `2.939`; all coordinate means,
coordinate second moments, and adjacent cross moments passed. The identity
scale and width changes did not repair the baseline failure. The evidence
therefore points to the permutation/order geometry as the target-specific
repair under this setup, while capacity and smaller initialization scale are
not supported as repairs.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Gaussian LR nomination | Both confirmation seeds pass exact-law screens | **Passed** for `1e-3` | Two confirmation seeds | Retain `1e-3` as Gaussian control warm start and replicate only if needed | Globally optimal LR or default LR |
| Banana permutation repair | Both confirmation seeds pass exact-law screens | **Passed** for root-preserving at `5e-4` | Two screening and two confirmation seeds; other permutation/order variants untested | Use root-preserving as a target-specific banana candidate for an independent replication | Universal superiority or architecture default |
| Identity-biased initialization | Exact-law screens | **Failed** in screening | Only two seeds | Do not pursue this arm without a new hypothesis | Initialization is universally harmful |
| Width 64 | Exact-law screens | **Failed** in screening | Only one width and same training budget | Do not promote capacity expansion from this result | Capacity is irrelevant generally |
| HMC readiness | Downstream sampler validity | Not evaluated; proposal policy remains unpromoted | No HMC run in this campaign | Require a separate HMC plan after replication | Posterior correctness |

### Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Gaussian `1e-3` and banana root-preserving candidate passed confirmation; other tested arms failed or were not nominated |
| Statistically supported ranking | None; losses, ESS, and standardized discrepancies are descriptive and two-seed estimates |
| Descriptive-only differences | Root-preserving banana passed while baseline, identity-biased, and width-64 screening arms failed; this is viability classification, not superiority |
| Default-readiness | Not supported |
| Next evidence needed | Independent multi-seed replication of the two target-specific candidates, followed by a separate downstream/HMC validity plan |

## Red-Team Note

The strongest alternative explanation is that the root-preserving permutation
changes the optimization basin rather than representing an intrinsically
better transport. The campaign controls update count, batch, objective, and
LR selection, but it does not identify the causal mechanism. The weakest
evidence is any ranking among failed banana arms. A confirmation failure on
new seeds would overturn viability for the root-preserving candidate without
invalidating the harness.
