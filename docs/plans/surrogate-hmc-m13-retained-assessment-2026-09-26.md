# M13 retained-posterior assessment and simple comparator

This is an implementation stage of the approved September25 continuation, not
a new campaign. The question is whether the selected analytical DCTR-force
surrogate HMC samples the independently corrected M13 posterior materially
accurately. Data1704/T20 is an exploratory pilot. Data3701 remains untouched.

## Evidence contract and limits

Use the frozen full-support4096x8 endpoint for both HMC and random-walk
Metropolis (RWM). Use the saved129x129 grid with24 independent16384-particle
replicates as the independent reference. A normalized bilinear interpolation
of the posterior density defines the finite-box diagnostic law. Its first two
moments and quartile CDFs are integrated exactly for that interpolant. It is
not the full-support target; grid refinement, particle replication and saved
domain expansion quantify empirical uncertainty without proving a tail bound.

The existing margins remain means and SDs0.1referenceSD and six CDFs0.05.
Four retained chains are assessed only at1000 and2000 draws per chain, after
250-draw health chunks. Warmup and tuning verification are excluded. Promotion
requires finite values, endpoint/force health, movement, rank/folded R-hat<=1.01,
bulk/tail ESS>=400 and simultaneous MCSE-aware agreement. Practical point
agreement with wide intervals is formally inconclusive, not material failure.

For ten functions use batch-mean variance at10,20,40 batches per chain, the
nonnegative maximum including2V10-V20,2V20-V40 and the between-chain mean
variance. The SD uses its delta-method influence. These choices protect against
short-batch bias but still assume stationarity and sufficient mixing. A
Student-t36 critical value allocates alpha0.025 across ten functions and two
looks; the remaining0.025 is allocated to reference replication. Add absolute
fine/coarse and expanded-domain changes as empirical allowances, never rigorous
error bounds. Report the allowances separately from chain MCSE.

Before any retained draws, check interval coverage against the actual frozen
finite-grid law using256 independent four-chain ensembles for IID draws and
sticky-refresh probabilities0.5,0.2,0.05. Exact-law IID draws followed by an
independent refresh/hold sequence give known stationary correlated controls.
Use disjoint fixed seeds per condition. Check all ten functions jointly across
both looks; require the one-sided95% binomial lower bound on family coverage
to be at least0.95 in every condition. This is a scoped calibration check, not
a universal guarantee. Failure blocks interval qualification and triggers a
documented repair within the existing allocation; do not tune on its holdout.

## Baselines, assumptions and decisions

Construct direct finite-grid IID draws (reference mechanics/coverage), RWM
with Gaussian proposals (same-endpoint classical comparator), and independent
reference integration (posterior accuracy). Prior-scale starts exercise broad
initialization; nominal, intermediate and outer reference regions exercise
tails. Record conditional proposal/acceptance behavior for RWM. Do not infer
relative speed or superiority from a single pilot. Endpoint-gradient HMC is
excluded until its derivative is independently checked, as the parent plan
permits. Baselines cannot select HMC controls or tune posterior margins.

RWM proposal covariance starts at the prior covariance, then uses a separate
pilot covariance with10% prior shrinkage to ensure positive definiteness.
Adapt one scalar proposal scale during discarded warmup toward0.25 acceptance
(a tuning hypothesis). Freeze the proposal before four fresh retained chains.
Keep all warmup, report the proposal and acceptance, and apply the same retained
diagnostics. An unconverged comparator cannot establish HMC superiority.

Reference calculations and interval coverage are explicit CPU diagnostic
exceptions with GPUs hidden before TensorFlow import. RWM is a reference
comparator and may use the same CPU exception. TensorFlow/TFP and compiled
stable-signature numerical kernels remain required; no pfor or NumPy runtime.
These stages share the approved6000CPU/6000GPU baseline allocation. HMC retained
sampling retains its56000CPU/48000GPU cap, with no automatic extension.

Audit before execution: the baseline has the same likelihood, prior and data;
the independent reference uses fresh designs; repeated looks are prospectively
limited; practical discrepancies are distinguished from precision; finite-box
and single-pilot limitations are explicit. Failures of coverage, convergence
or material agreement reject that candidate or evidence claim, not the whole
surrogate-HMC direction. Corrupted provenance or wrong targets stop execution.

Exact commands, dependency hashes, seeds, wall time, CPU/GPU choice and result
paths will be recorded with the implementation before its first research run.
This plan is not an execution result or a posterior qualification.

## Concrete CPU launch and pre-run review

The dedicated launcher is
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_m13_assessment.py.
Stage review writes to
docs/plans/artifacts/surrogate-hmc-m13-assessment-development-20260926-02;
stage reference-coverage writes to
docs/plans/artifacts/surrogate-hmc-m13-assessment-reference-20260926-01.
Pass each location with --output and select the named --stage. Each launch
writes its command log, source hashes, commit, CPU-only exception, wall/CPU
receipt and result; development receipts belong to the6000CPU baseline pass.
The first focused review passed5 tests and consumed7.886398 CPU seconds.
Two additional independent checks cover exact binomial inversion and compiled
refresh/hold mechanics before the coverage launch.

The implemented inverse-CDF draw avoids a potentially quadratic XLA categorical
temporary. Analytic polynomial moments/CDFs and a nonuniform exact-law sampling
test passed. Reference normalization retains replicate evidence weights rather
than averaging normalized posteriors. The domain linkage and source/grid hashes
are rechecked. The skeptical audit found no remaining target mismatch for this
finite-box diagnostic; full-support tail accuracy remains unproved.

## Local memory repair, before retained sampling

Reference-coverage01 was deliberately terminated after its XLA upper-bound
search reached about35 GiB RSS. It consumed785.162074CPU seconds; the two
completed conditions had family-coverage lower bounds0.981604 and0.975613.
This was a diagnostic implementation/resource failure, not evidence against
surrogate HMC or the interval rule. All original logs and charges are retained.
An explicit vector binary search now has O(draws+categories) live storage,
with fixed logarithmic depth. An independent upper-bound parity test includes
zero-mass cells and boundary values. Coverage now records peak RSS after each
batch and stops above8GiB. The same seeds, draws, conditions, margins and
coverage threshold remain frozen. Retry reference-coverage02 uses the same
dedicated launcher and a fresh versioned output root. Development and retry
charges are subtracted automatically from the6000CPU baseline allocation.

Budget clarification: the6000CPU allowance is shared with materiality-control
attempts01/02, whose1274.239627CPU seconds must also be deducted. The launcher
now reads those original receipts as well as every assessment receipt. The
materiality-repair attempt belongs to the separate10000CPU repair allocation.
No stage receives an extra6000CPU allocation. Run development03 next, followed
by reference-coverage02 only if all focused checks pass.
