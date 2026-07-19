# SSL-LSTM NeuTra Direct Visual Validation Plan

Date: 2026-07-18  
Status: completed; visual package passed and confirmation remained closed  
Tier: Tier 2 material research engineering

## Question and Scope

Can the admitted scalar SSL-LSTM NeuTra arms be displayed in a way that makes
initialization robustness and forecast agreement directly inspectable, without
mistaking attractive plots for posterior correctness? The program uses only
the already opened, permanently confirmation-excluded 64-draw prefix of each
admitted G/H chain and the frozen ten-step forecast operator.
It does not retrain NeuTra, run HMC, or open the closed G/H confirmation gate.

The scalar target is the frozen Phase-7 target with `latent_dim=1`,
`hidden_dim=1`, `observation_dim=1`, four chains, and 512 retained draws per
chain. This plan reads only indices `0:64` from segment 0, which the Phase-8
pilot permanently excluded from confirmation. It does not deserialize segment
1 or inspect indices `64:512`. G and H are independent peer replications, not
a posterior oracle.

## Research Intent Ledger

| Item | Frozen statement |
| --- | --- |
| Main question | Are the two admitted arms visually and predictively similar over the fixed ten-step forecast, and do chains from distinct starts show stable behavior? |
| Candidate/mechanism | Fresh-G and fresh-H frozen NeuTra transports followed by the same exact-corrected forecast operator. |
| Expected failure mode | Chain-dependent trajectories, fan-chart separation, non-finite forecasts, or moment differences larger than their declared uncertainty. |
| Primary promotion criterion | All required artifacts are produced; source/provenance hashes and forecast invariants pass; visual claims are limited to the declared scope. No plot alone promotes posterior correctness. |
| Promotion veto | Hash/provenance mismatch, wrong tensor orientation, non-finite output, missing chain, invalid terminal covariance, seed overlap, or failed LaTeX/build/test check. |
| Continuation veto | Corrupt or unauthenticated retained artifacts, inability to map retained samples through the frozen transport, or resource cap exceeded before a valid receipt. |
| Repair trigger | Any isolated orientation, plotting, numerical, or documentation failure triggers a focused repair and rerun; it does not reject the predictive-validation direction. |
| Explanatory diagnostics | Chain traces, launch-to-launch distances, fan-chart overlap, moment bands, per-horizon standardized differences, and runtime. |
| Must not conclude | Posterior correctness, HMC convergence beyond the admitted screens, G/H equivalence or material difference, sampler superiority, model adequacy, or default/production readiness. |

## Evidence Contract

| Role | Contract |
| --- | --- |
| Scientific question | Do independent launch chains and forecast paths provide an intuitive view of stability and predictive agreement? |
| Comparator | Fresh-G versus fresh-H, with the same authenticated scalar forecast operator and disjoint independent innovation banks. |
| Primary pass/fail | Artifact integrity, exact shape/dtype/provenance checks, finite forecast paths, valid covariance status, and successful figure/LaTeX/test generation. |
| Veto diagnostics | Any provenance/hash mismatch, chain omission, tensor orientation error, non-finite path, covariance failure, seed overlap, or stale source signature. |
| Explanatory only | Visual overlap, trace mixing, fan width, moment differences, and runtime. They cannot establish correctness or rank arms. |
| Nonclaims | No posterior oracle exists; no parameter-space agreement claim is made, and no visual result opens confirmation. |
| Preserved artifact | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation/visual-validation-result.json`, PNG/PDF figures, and this plan's result note. |

## Mathematical and Plot Contract

1. Parse only segment 0 using the manifest-authenticated `tf.io.parse_tensor`
   route. Its persisted shape is `(256, 4, 4)`; select indices `0:64` and
   transpose to `(chain, draw, parameter)` only after checking the manifest.
   Hash-check segment 1 without deserializing it. The four
   original chain starts are the chain axis, not four independent posterior
   samples to be pooled without labels.
2. Map transformed samples through the exact frozen `forward_z_to_theta_batch`
   transport. Record the transport, target, tensor, and manifest hashes.
3. Produce four figure panels: (a) transformed-coordinate traces by chain for
   all four coordinates, (b) selected physical-parameter traces by chain, (c)
   ten-step predictive fan charts with chain-colored means and 5--95% bands,
   and (d) horizon-wise mean and log-variance differences with the existing
   simultaneous-region boundary if the target adapter receipt is available.
4. Use an independent innovation bank for each arm and never reuse calibration
   innovations. Record roots, bank signatures, and output hashes. The fan
   chart is a Monte Carlo visualization, not a credible interval for the true
   posterior predictive law.
5. Compute descriptive per-horizon means, variances, and differences. If the
   frozen target-adapter receipt is used, report its formal region status
   exactly; do not recompute or alter its thresholds in the plotting script.
   The plot may add explanatory Bonferroni Gaussian bands estimated from
   chain-preserving 16-draw block influences. Those bands are visibly labeled
   approximate and cannot emit an equivalence/material-difference decision.
6. Run a synthetic-data coverage smoke using the existing controlled-law
   harness only if its locked receipt and independent seed domain are present.
   Label this as controlled calibration evidence, not retained-chain evidence.

## Execution Phases

### Phase 1: Artifact and source audit

Read the Phase-7 public receipt, both private manifests, frozen transport
payloads, forecast configuration, and target-integration receipt. Verify hashes,
segment-0 shape `(256,4,4)`, four-chain identity, segment ordering, and no overlap with
calibration/evaluation innovation seeds. Entry condition: Phase-7 admission is
present and its confirmation-closed status is unchanged.

Handoff: an audit JSON records every input hash and passes all integrity gates.

### Phase 2: Plot runner and focused tests

Implement one bounded runner under `docs/benchmarks/` plus focused tests. Keep
the runner read-only with respect to retained/private inputs and write only to
the direct-visual-validation artifact directory. Use TensorFlow/TFP for the
forecast path; use Pillow only for deterministic PNG/PDF rendering. Default serious execution is
GPU/XLA; a CPU-hidden reference mode is allowed only for smoke/debug and must
be labeled as such.

Handoff: unit tests cover tensor orientation, chain labels, finite fan/moment
outputs, seed disjointness, and fail-closed provenance checks.

### Phase 3: Bounded visual execution

Run the four-panel package on the permanently excluded 64-draw pilot prefix
with one independent bank per arm, a declared wall cap of 900 seconds, and
sequential stop after the first complete valid package. Save PNG and PDF
figures, a machine-readable receipt,
and the exact command/environment/device/JIT manifest. Do not access any new
HMC or NeuTra training artifact.

Handoff: receipt status is `PASSED_VISUAL_PACKAGE` or a repair result records a
specific failed gate; G/H confirmation remains closed in either case.

### Phase 4: LaTeX integration and review

Add a self-contained subsection to Chapter 28a with figure captions, numerical
tables, test results, source citations already used by the chapter, and the
claim boundary. Build `docs/main.tex` with `latexmk -pdf`; run a focused text
audit for stale numbers, unsupported claims, missing labels, and figure paths.

Handoff: PDF builds cleanly and the chapter numbers/hash references agree with
the receipt and result note.

## Skeptical Pre-Execution Audit

| Audit question | Finding |
| --- | --- |
| Wrong baseline? | No. The comparator is the two admitted, independently seeded G/H arms, not a weak baseline or oracle. |
| Proxy promoted? | No. Plots and trace/fan overlap are explicitly explanatory; only integrity and declared formal-receipt statuses can pass. |
| Missing stop? | No. A 900-second cap, one complete package stop, and fail-closed artifact veto are explicit. |
| Unfair comparison? | No. The same forecast config and disjoint independent innovation roles are required for both arms. |
| Hidden assumptions? | Tensor orientation, pilot-prefix boundary, transformed-to-physical mapping, scalar dimensions, and terminal covariance status are frozen. |
| Stale context? | No. The Phase-7 receipt and 2026-07-18 target-integration receipt are named inputs; no posterior-oracle assumption is used. |
| Environment mismatch? | GPU/XLA provenance is required for serious execution; CPU-hidden mode is labeled reference only. |
| Do artifacts answer the question? | Yes for intuitive stability/predictive visualization and artifact integrity; no for posterior correctness or G/H confirmation. |

Audit decision: `PASS_FOR_BOUNDED_VISUAL_PACKAGE_ONLY`.

Audit amendment: the first draft proposed plotting and forecasting all 512
retained draws per chain. That would have inspected the 448-draw suffix reserved
for future confirmation. Before implementation or sample access, the plan was
repaired to deserialize only segment 0, use only indices `0:64` already opened
by the Phase-8 pilot, and hash-check all other files without reading their
tensor values. This preserves the closed confirmation boundary.

Environment amendment: the TensorFlow/TFP `tfgpu` environment does not contain
matplotlib. The runner therefore uses the already installed Pillow library for
deterministic PNG and PDF rendering. This changes no forecast, statistic,
threshold, seed, or evidence interpretation and avoids a dependency change.

## Stop Conditions and Nonclaims

Stop on any input-hash drift, private-manifest mismatch, wrong tensor shape,
chain/segment omission, seed collision, non-finite forecast, invalid terminal
covariance, plotting failure, LaTeX failure, or resource cap. A failed plot
package is an engineering repair signal. It is not evidence against NeuTra or
predictive validation. No result from this plan may be called an oracle,
posterior proof, sampler ranking, or model-adequacy result.

## Execution Close

The bounded run completed with decision
`PASSED_VISUAL_PACKAGE_CONFIRMATION_CLOSED`. Five PNG/PDF figure pairs and the
machine receipt are preserved under the declared artifact directory. Focused
tests and the full LaTeX build passed; no confirmation-suffix tensor value was
read. The detailed numerical result is in
`bayesfilter-ssl-lstm-direct-visual-validation-result-2026-07-18.md`.
