# Original Zhao--Cui code reference repair: completed execution

The linear-preconditioned author TT solver now runs through a reproducible
Octave diagnostic harness,
including fixed-target runs on the exact saved T=20 observations. Predator--prey
has a finite, useful diagnostic estimate. The tiny SIR configuration has severe
importance-weight collapse and is unsuitable as an accuracy reference. The
repair does not supply an analytical likelihood score.

## Published-result comparison: not reproduced

Correction and source recheck, 2026-10-04: this work completed a reduced
execution diagnostic, not reproduction of the published experiments. The SIR
result does not establish that Zhao--Cui's method or published results are wrong.
Even its cause is not isolated: inadequate approximation effort is plausible,
but source/adapter fidelity must also be established by a matched reproduction.

The checked primary publication is Zhao and Cui, JMLR 25(244), 2024,
[Sections 6.3--6.4, pp. 38--42](https://jmlr.org/papers/volume25/23-0743/23-0743.pdf).
The local copy is
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`.
The publisher page was checked on 2026-10-04 and displayed no correction notice;
this is a bounded comparison, not a comprehensive errata or literature survey.

| Quantity or choice | Published experiment / checked author driver | Executed diagnostic |
|---|---|---|
| SIR maximum TT rank | Paper: 10, 20, 40; text reports ranks above 20 needed for accuracy | 4 |
| SIR ALS iterations | Paper: 5; pinned main/low options: at most 8/2 | 1/1 |
| SIR base fitting samples | Pinned driver: 5,000 | 64 |
| SIR smoothing draws | Pinned driver: 10,000 | 64 |
| Data | Paper/author synthetic experiment | Different saved BayesFilter observations |
| PP unknown parameters | Jointly infer six parameters | Fix all six at the current target point |
| PP successful preconditioning route | Paper: nonlinear preconditioning, rank 20, five ALS iterations; about 40% joint ESS at T=20 | Linear-preconditioned full_sol route, rank 4, one iteration |
| Reported quantities | Sections 6.3--6.4 show joint-posterior ESS and state paths | Added log-evidence diagnostic and bootstrap comparison |

Paper anchors are Section 6.3, Figure 15 and the paragraph spanning pp. 39--40;
Section 6.4, Figure 17 and the discussion on p. 42. Author-code anchors under
`third_party/audit/zhao_cui_tensor_ssm_p10/source/` are
`eg3_sir/mainscript.m:39-64` and `eg4_predatorprey/mainscript.m:45-93`.
The actual reduced consumer is
`docs/benchmarks/run_zhao_cui_reference.py:121-130`; it hardcodes one ALS
iteration and calls full_sol_reference, never pre_sol. Raising its rank alone
would still not reproduce the paper. The paper and pinned drivers also differ
in their ALS settings; a reproduction must declare which it follows.

The narrow statistic correction remains valid: if a raw log importance weight
is w, averaging w is a different quantity from log(mean(exp(w))). However, the
optional fifth output lml at `models/full_sol.m:191-204` is not requested by
the SIR or PP example drivers (`eg3_sir/mainscript.m:64`;
`eg4_predatorprey/mainscript.m:88,92`). Correcting that diagnostic for our
likelihood use does not establish a defect in the paper's filtering, ESS or
state-estimation results. These sections publish no matching log-likelihood
table for our saved observations.

The original-code reference is therefore not yet validated against the
publication. The next scientific task is an explicitly matched reproduction
of the published ESS and trajectory results, including the PP pre_sol route,
before adapting it into a likelihood/score reference for our data. No new
experiments were run for this source recheck.

## Actual saved-data results

Both cases use base N=64, 64 smoothing draws, rank at most 4 and one ALS pass.
The source solver can enlarge its proposal sample set internally. These are
mechanics settings, not tuned numerical defaults. The following comparison is
descriptive; one source seed cannot support a statistical accuracy ranking.

| Model | Corrected logmeanexp | Legacy mean log weight | TT log normalizer | ESS | Saved bootstrap log likelihood | Difference |
|---|---:|---:|---:|---:|---:|---:|
| pp | -97.335237 | -97.435313 | -97.425252 | 51.379/64 | -97.342974 | +0.007737 |
| sir_austria | -4253.054415 | -6068.379866 | -1228.154200 | 1.000/64 | -678.077466 | -3574.976949 |

The saved bootstrap comparator has 524,288 particles per replication and the same
observation hash and nominal physical parameter point. The source probes use
float64; the saved PP float32 parameters differ by at most 2.384186e-8 (the SIR
parameters match exactly). `comparison-parameter-audit.json` records both points.
This precision boundary is another reason to treat the comparison as descriptive.
The bootstrap calculation is approximate, not an exact oracle. Its original summary, replication
counts and reported MCSE are preserved in `saved-bootstrap-reference-summary.json`
and `saved-reference-comparison.json`. No new bootstrap run or speed comparison
was made. The PP difference is small in this one run; statistical agreement and
convergence of the source estimate have not been established. The SIR difference
and disagreement with its own TT normalizer require a larger, calibrated TT and
smoothing study before quantitative use.

## Problems found and repairs

1. The model callbacks are MATLAB Live Scripts in ZIP/XML containers. The bridge
   extracts all 19 callbacks into a fresh derived tree and records source and
   output hashes.
2. Octave lacks the used `mvnrnd` and rejects several MATLAB property annotations.
   A Cholesky Gaussian sampler and syntax-only class overlays run in the derived
   tree. RNG results are reproducible within Octave, not bitwise MATLAB parity.
3. `models/full_sol.m:192-205` reports the mean log importance weight, which is
   wrong relative to the log evidence estimator. The derived class retains that
   old value but also reports stable logmeanexp, all raw weights, finite fraction,
   normalized-weight ESS and invalid status on nonfinite weights.
4. SIR smoothing evaluated Gaussian PDFs before taking logs, producing numerical
   zeros. Closed-form Gaussian log densities now evaluate the same target during
   smoothing. TT fitting continues through the author source. Healthy-density
   parity errors were 1.78e-15 for author PP and 7.11e-15 for author SIR; healthy PP
   results before and after the stable-log change are identical.
5. Author PP integrates a parameter prior, uses different physical parameters,
   and evaluates the fourth RK stage at a half-step. The fixed-target adapter
   conditions on the current physical parameters, maps the source ODE chart, and
   uses the current full fourth stage. This is explicitly an
   `extension_or_invention`, not a reproduction of the author's PP experiment.
6. The current SIR consumer already preserves the author's half-step convention.
   At zero log scales its Gaussian density callbacks align with the source. Both
   models receive byte-preserved saved datasets from Git (the originals are
   omitted locally by sparse checkout); the TensorFlow observation hashes are
   verified before injection.
7. The author PP proposal has bounded support. The fixed-target PP extension
   uses the author's available unbounded `AlgebraicMapping(1)` option. Source
   reproduction still uses its original bounded domain. Support and tail coverage
   remain accuracy questions, even when all arithmetic is finite.

For a smoothing path z drawn from proposal q, write
`w(z) = log p(z,y) - log q(z)`. Then `mean(exp(w))` estimates
`integral_[q>0] p(z,y) dz`, and logmeanexp is its logarithm. Full evidence requires
coverage of the target support. When q has support within the posterior,
`E_q[w] = log p(y) - KL(q || p(z|y))`; thus the legacy mean log weight is an
expected-log-weight quantity, not log evidence. Taking the log of a Monte Carlo
mean also introduces finite-sample bias. Correcting the statistic repairs none
of the proposal-accuracy or support problems by itself.

The source SIR generative sampler clips susceptible states after adding noise,
whereas its transition callback evaluates an unconstrained Gaussian. The
fixed-target comparison evaluates the latter, matching the current likelihood;
the retained source sampler affects the fitting proposal. The author-data smoke
therefore reproduces a source convention, not a proof that all source callbacks
form a single generative law at the clipping boundary.

## Verification and terminal review

The T=20 call chain is `NonlinearSQMCSpec.model` to the current canonical model for
TensorFlow probes, and derived `full_sol_reference` to the pinned source `TTSIRT`
for fitting/smoothing. Resolved Octave paths are saved per case. Five deterministic
state probes cover initial, perturbed, low-state and higher-infection/predator
conditions. Transition means and initial, transition and observation log densities
agree with maximum absolute errors 0 for PP and 2.328306e-10 for SIR (threshold
1e-8). Initial laws and noise covariances are unchanged; the adapter fixes model
parameters explicitly. This is executable pointwise model/call-chain evidence,
not an all-input theorem or posterior-quality test.

Independent Python calculations from every raw log weight agree with the Octave
corrected summaries. Every smoothing weight is finite in the final runs. The
complete pinned source tree has identical SHA-256 fingerprints before and after
execution. Eleven focused regression tests pass (final check: 2026-10-04). Compilation,
extraction-only execution and a deliberately rejected preparation also pass;
`validation/validation.json` and its logs preserve the checks. The final small
manifest-only edit adds structured preparation failures and does not change the
numerical path used in the retained experiments. CPU-only TensorFlow eager probes
are deliberately small reference exceptions; Octave is CPU-only. No production
backend or default was changed. Archived derived source retains the author
whitespace, so its diff-check warnings are preserved rather than changing
bytes covered by the recorded hashes; new code and prose pass the scoped check.

The reviewed plan is `docs/plans/zhao-cui-reference-repair-20261003.md`. Failed
attempts, exact commands, environments, seeds, source/extractor hashes, `.mat`
path samples and logs remain under the versioned artifact root. Octave cases
consumed 89.716 of 900 allowed wall seconds; 810.284 seconds were unspent.
The early `author-reference-01` failure is a harness-introspection failure even
though the initial broad classifier called it `numerical_or_source_route`.
The attempt ledger preserves that correction without overwriting the old output.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete reference bridge | Source execution, provenance and corrected statistics pass | No final nonfinite/provenance failure | MATLAB parity and full source numerical behavior | Use the documented diagnostic runner | Source correctness or production readiness |
| Accept fixed-target wiring at tested point | Saved hash and five-probe density/transition parity pass | PP original target mismatch explicitly adapted | Probe coverage and TF32 numerical differences | Keep source and adapted results distinct | Unmodified author PP equivalence |
| Keep PP as a diagnostic comparator | Corrected value -97.335237, ESS 51.38/64 | No arithmetic veto | One TT/smoothing seed, no source MCSE | Replicate and vary rank/N before accuracy admission | Oracle or statistically supported accuracy |
| Exclude this SIR configuration from accuracy reference use | Reference quality unestablished; ESS 1/64 | No model/harness invalidation; severe weight collapse | Low rank, one ALS pass and full-path importance variance | Calibrate TT quality and validate sequential normalization with repeated runs | Rejection of Zhao--Cui or LEDH research direction |
| Leave score unavailable | No source analytical score exists in this bridge | Claiming one would be wrong | Independent score construction and convergence | Separate score-reference work | Analytical or HMC-ready score |

| Inference item | Finding |
|---|---|
| Hard veto screen | Final finite arithmetic, source immutability and model parity pass; earlier failures preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | All numerical differences in the table; particularly PP closeness to bootstrap |
| Default readiness | Not evaluated; production code unchanged |
| Next evidence needed | Same-data scope-specific TT/sample calibration, independent repetitions and uncertainty; a separate score construction |

The terminal skeptical review passes the engineering repair and rejects any
promotion of these runs to a converged value/score oracle. The strongest
alternative explanation for SIR is inadequate TT approximation at deliberately
small settings, compounded by full-path importance variance. A larger calibrated
TT route whose sequential normalizers and repeated independent importance
estimates agree with the same-data reference could overturn that assessment.
The weakest evidence is one source seed. No continuation veto invalidated the
method; the bounded mechanics plan is complete, and accuracy calibration is the
next research task rather than an unfinished step of this plan.

## Reproduce

Run `docs/benchmarks/run_zhao_cui_reference.py --output-root <fresh-directory>`
for author-generated T=2 smoke data. For the saved-data adapter use
`--mode fixed_target --horizon 20`, supply both saved `screen-01/.../dataset.json`
paths through repeated `--dataset` arguments, and set a fresh output root.
The exact commands are in the final manifests. CPU probes require the existing
`tftwogpu` environment; Octave 6.4 is the source audit environment. The runner
reads absent sparse-checkout datasets from the recorded Git commit and refuses
hash, horizon, or target mismatches. It never silently slices or regenerates data.
