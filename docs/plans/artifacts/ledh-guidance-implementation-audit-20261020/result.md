# Additional implementation audit, 20 October 2026

Two additional executable defects are confirmed: misleading guide-validity
flags and an incomplete paired-result validator. There are also gaps in
diagnostic retention and source provenance. The major missing capabilities
remain actual KDM training, full controlled-SMC twisting, and a learner with
more than a few adapter parameters. These are different findings: the first
two are implementation defects; the restricted learner and proposal are
explicit pilot choices that do not implement the larger research proposal.

This audit covers the executed SIR physical-proposal training consumer in
`ledh-refactor-with-policy-fix`, base `4d6915cc66`. It does not certify every
BayesFilter algorithm, the original-author replications, or other models.
No runtime implementation, numerical setting, or original result was changed.
No training, new heldout simulation, GPU experiment, or selection was run.

## Confirmed additional defects and omissions

### 1. The physical branch reports a bad guide as valid

`bayesfilter/highdim/ledh_canonical_score_tf.py:424` initializes
`guided_proposal_valid=True`. The pre-flow guide branch updates it, but the
physical proposal branch at line558 updates a separate `proposal_valid`.
Consequently `backward_guide_valid` and `guided_proposal_valid` remain true
even when the guide consumed in the physical branch is invalid.

The actual consumer reproduction uses the existing healthy two-dimensional
fixture with guide precision `-0.01 I` under the default PSD-only guide
policy. The total proposal precision remains SPD, so this isolates policy
validity from a broken factorization. Direct guide evaluation returns false.
At both filter times the trace instead reports:

| Field | Returned value |
| --- | --- |
| `backward_guide_valid` | true, wrong for the consumed guide |
| `guided_proposal_valid` | true, wrong for the physical guided proposal |
| `physical_proposal_valid` | false, correct |
| `program_valid` | false, correct |

The returned likelihood is negative infinity. Thus the final rejection works;
this is a wrong diagnostic, not acceptance of an invalid filter. It could
misdirect failure localization. Evidence: [validity-reporting.json](validity-reporting.json),
reproduced by [check_validity_reporting.py](check_validity_reporting.py).
Repair: propagate the actual physical guide/proposal flags into the trace,
and regression-test both valid and rejected guides at the consumer endpoint.

### 2. The result assembler accepts incomplete or unpaired replications

`bayesfilter/score_study/physical_proposal_diagnostic_campaign.py:327`
averages whichever rows are present per dataset/arm. It does not enforce the
declared replicate key set, uniqueness, or paired particle seeds. Its
comparisons therefore have a weaker validity check than the experiment plan.

| Input copied from the saved result | Rows | Defect | Comparisons reported valid |
| --- | ---: | --- | ---: |
| Original | 544 | none | 42/42 |
| One trained-cSMC replicate removed | 543 | missing replication | 42/42 |
| One trained-cSMC replicate duplicated | 545 | duplicate replication | 42/42 |
| One trained-cSMC particle seed changed | 544 | broken particle pairing | 42/42 |

The original file has exactly the expected keys, no duplicates, no invalid
rows, and no mismatched particle seeds. This defect does not invalidate that
file or explain its negative training result. The mutations are integrity
counterexamples, not new scientific comparisons. Evidence:
[result-assembly.json](result-assembly.json) and
[check_result_assembly.py](check_result_assembly.py).

This is a regression in coverage: the earlier
`moment_training_analysis_diagnostic.py:36` rejects incomplete/duplicated
grids, while the new campaign implements a different assembler. Existing
tests of that earlier analyzer do not protect the executed consumer.
Repair: validate the frozen dataset/arm/replicate grid and paired seeds before
computing means or intervals. Missing data must produce an explicitly
incomplete result, not a valid paired comparison.

### 3. The eight-file hash check is not complete executable provenance

The campaign's explicit source list at
`physical_proposal_diagnostic_campaign.py:81` omits executed dependencies,
including the actual higher-moment implementation, shared reset implementation,
Gaussian-mixture conditioner, SIR simulator/model, and multivariate cSMC fitter.
The recorded correction wrapper calls `higher_moment_shape_jvp` from
`higher_moment_contract_e.py`; that unrecorded implementation is currently
dirty relative to the recorded commit.

All eight recorded hashes still match. This proves those eight files match,
not that the whole executable dependency closure matches the original run.
The current dirty state does not establish when an omitted file changed, and
this audit does not allege that it changed after training. Evidence:
[source-coverage.json](source-coverage.json), with seven inspected missing
dependencies, and [check_source_coverage.py](check_source_coverage.py).
Repair: preserve the relevant dirty source snapshot/diff and dependency hashes
with the run. No new authorization-token machinery is needed.

### 4. A computed conditioning margin is discarded

`gaussian_mixture_conditioning_tf.py:97` returns the minimum relative SPD
margin across the Gaussian product computation. Both
`ledh_physical_proposal_tf.py:75` and `ledh_backward_guide_tf.py:214` unpack
that margin into `_`. Its validity flag survives; the margin does not reach
the filter trace or campaign record. This is incomplete numerical
observability, not a demonstrated wrong proposal density. Retaining it is a
routine observability repair; an inactive-guide sentinel must be distinguished
from an actual conditioning measurement.

## What the current learner still does not implement

| Intended capability | Executed implementation | Verdict |
| --- | --- | --- |
| Train the replicated KDM model for downstream moments | `SIRGuideFitter` fits positive RBFs to a cSMC-derived future message; the campaign never calls the replicated neural KDM trainer | KDM integration incomplete; calling this original KDM training is wrong |
| Controlled-SMC twisted intermediate distributions and their compensating weights | A fitted future message changes the proposal, then `log f + log g - log q` restores the filtering target | Valid proposal guidance; full twisting is not implemented in this consumer |
| Learn flexible control/message parameters | `make_guide_adapters` learns one strength and two regional shifts, pooled or per time: 3 or12 scalars | Restricted adapter optimization, not full policy/network training |
| Adapt proposal allocation | Both guided arms fix the physical-component probability to0.5 | Allocation cannot respond to the observed ineffective flow component |
| Use the earlier three-density covariance mixture | This consumer uses two densities: the LEDH pushforward and the physical conditional proposal | The scalar probability here is not the earlier three-component beta vector; that allocator is not connected to this experiment |
| Establish improved score estimation across models/scopes | Terminal first/raw-second-moment loss at one SIR parameter, T5/N64, FP64 | Score improvement, parameter-neighborhood transfer, longer horizons and other models remain unevaluated |

Call chain: `PhysicalCampaign.__init__:38–79` constructs `SIRGuideFitter`,
the three-scalar adapters, and the dynamic physical-proposal kernels;
`evaluate:123` passes their coefficients into the shared executor;
`train:275` optimizes only those adapter tensors. The fitter's imports and
`fit:191` call the multivariate cSMC engine and its own RBF export, not the
replicated KDM source-training classes. The parameter signatures of the
adapters at `moment_training_tf.py:47` expose none of the missing controls.
These restrictions were already substantially visible in the previous
diagnosis and are reconfirmed here, not presented as newly discovered algebra.

The mathematical squared-error identity for simulated labels remains correct.
It does not imply that a restricted finite-particle family contains a better
estimator, or that the implemented search finds one. Frozen guide coefficients
also define a legitimate finite value program: differentiating that program
does not require differentiating a guide refit that the program never performs.

## Checks that passed and limits of this audit

The focused suite passed **23 tests in46.65 seconds**, with two TFP dependency
deprecation warnings. It covers the actual physical consumer's analytical
derivative against finite differences, both component densities, beta=0
parity, beta=1 cancellation, dynamic guide wiring, adapter/loss behavior and
SIR guide mechanics. See [cpu-tests.log](cpu-tests.log).

Source and algebra checks found no additional error in the physical mixture
denominator, Gaussian-product moments, sample/density tangents, or the use of
independent simulated labels. The shared Gaussian observation fallback already
includes dR; the stale contrary model comment is not the current computation.
The reset's numerical-validity flag is consumed downstream, and the higher-
moment validity flag also rejects the whole calculation. Neither suspected
missing guard was a defect. The existing tests do not establish correctness
outside their fixtures, GPU/XLA parity for this audit, native KDM integration,
or scientific improvement.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Confirm two additional executable defects | Consumer/assembler counterexamples reproduced | No false acceptance by the filter itself; result validator misses malformed rows | No observed corruption in the saved original | Repair flags and result validation with consumer regressions | These caused the failed training |
| Record diagnostic/provenance omissions | Executed call sites and manifest coverage inspected | No new numerical veto demonstrated | Full historical identity of omitted dependencies | Retain computed margins and complete source provenance before another run | Recorded eight-file parity binds every dependency |
| Keep the larger training proposal incomplete | Actual learner/consumer path traced | Prior candidate remains unpromoted | Benefits of a richer learner and actual KDM integration untested | Implement and validate the specified learner under a fresh scoped plan | The mathematical idea is disproved |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | The invalid-guide fixture is correctly rejected overall; malformed result copies are not rejected by the assembler |
| Statistically supported ranking | No new ranking assessed |
| Descriptive-only differences | No new stochastic performance comparison |
| Default-readiness | Not established or changed |
| Next evidence needed | Focused repair regressions first; a separate fresh-data campaign for any scientific claim |

Red-team conclusion: a valid restricted pilot can fail empirically without
being an incorrect importance sampler. Conversely, passing helper tests does
not establish that a new consumer preserves diagnostics or evidence checks.
The strongest remaining limitation is scope: this is not a repository-wide
audit or an independent rederivation of every UKF/GenUT/OT primitive. A repair
must preserve the present correct mixture and total-derivative identities.

## Reproduction and resources

Workdir: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
Interpreter: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
CPU checks set `CUDA_VISIBLE_DEVICES=-1`, `TF_NUM_INTRAOP_THREADS=2`,
`TF_NUM_INTEROP_THREADS=2`, `TF_CPP_MIN_LOG_LEVEL=2`, and `PYTHONPATH=.`.
The focused test command is `python -m pytest -q
tests/highdim/test_ledh_physical_proposal_tf.py
tests/highdim/test_moment_training_tf.py
tests/highdim/test_sir_backward_guidance.py`.
Each `check_*.py` in this directory is directly executable from that workdir;
logs and structured JSON are preserved beside it. The two TensorFlow probes
record function wall time, excluding interpreter/import startup. The source
coverage probe uses only the standard library. No GPU charge; parent remaining
budget stays23826.197895057 GPU-process seconds. No process remains active.
