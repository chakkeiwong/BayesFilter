# M13 prospective control selection within the approved master continuation

The question is whether the full geometric numerical-protection family can be
frozen for the data1704/T20 surrogate-HMC pilot. The endpoint is held fixed;
the canonical analytical FP32 DCTR field is a proposal field and is not claimed
to be the derivative of that independent importance endpoint. User approval of
40 additional CPU-worker hours covers this stage; no Claude review is requested.

## Evidence contract and prospective defaults

Use N4096, K2048, canonical Contract E, FP32/TF32/GPU/XLA. Calibration data1705
and validation2705 are independent frozen streams, with force design partitions
12 and13. Data1704/design9 is the subsequent exploratory pilot; data3701 is
not evaluated. Changing this numerical scope requires a new selection.

The four controls are the pre-existing geometric family multipliers
1,.1,.01,.001. Their provenance is the earlier protection calibration family,
not evidence of optimality for M13. Their ridge/damping/cap changes must pass
the dedicated non-harm check below. All curve arms and original status are kept.

For every arm, evaluate four nominal corners (+/-.25,+/-.25) and four stress
corners (+/-3log2,+/-3log2). The nominal starts reuse the tested four-chain
connection geometry; they test local numerical behavior, not posterior coverage.
Stress corners test all four combinations of small/large process and observation
noise. Any unflagged invalidity invalidates the experiment. Flagged stress
failures are retained; a stronger arm may not introduce a failure where its
adjacent weaker comparator is healthy. Both arms must be healthy nominally.

Construct matched proposals with the actual repository endpoint-corrected HMC
kernel, prior-scale affine coordinates, L2, normalized steps .05 and .2, and
seeds (2026092501,100)/(2026092501,101). This produces16 acceptance probabilities
per arm. These steps span a small and the planned initial tuning step; they
are hypotheses, not tuned settings. They prevent selection from depending only
on the old .002 mechanics step. Common momentum/random streams make the arm
comparison paired. A single rejected proposal need not move; movement is a
later chain-level requirement. Nonfinite endpoint/energy, divergence or force
fallback makes that probe arm ineligible; do not hide these behind zero alpha.

Select the first adjacent pair whose maximum absolute alpha difference is at
most .01 on both independent datasets, with the stated health/stress screens.
The .01 probability allowance is prospective and follows the approved
continuation's materiality criterion. This small deterministic probe design
does not provide a probabilistic bound over parameter space. It only nominates
numerical controls for the downstream sampler test. Raw likelihood, score and
filter moment differences remain explanatory diagnostics. There is no .001
filter-statistic stopper. The endpoint and model law never change between arms.

Primary outcome: repository-issued scope/source/evidence-bound control selection.
Promotion vetoes: failed paired materiality or nominal/proposal health checks.
Continuation vetoes: unflagged invalidity, wrong data/law, source corruption.
Repair trigger: no adjacent passing pair; inspect the measured curves and use
the approved bounded repair allowance with fresh independent partitions.
No posterior convergence, exact score, superiority, or all-model validity is
concluded here. Downstream posterior/reference and random-walk comparisons
remain required by the parent continuation.

## Budget, artifacts and skeptical audit

Reserve4000 CPU-worker/4000 GPU seconds from the6000/6000 controls/baselines
allocation, including retries. Worker cap3800 CPU/3800 GPU seconds; focused
CPU tests cap120 seconds. Parent overhead consumes the same reservation.
Remaining2000/2000 in this allocation is for the cheap baseline/reference
report. Public tuning, retained sampling and repair have their own parent
allocations. Each attempt is a new versioned directory.

Command (escalated GPU access; tftwogpu environment):

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-materiality-controls --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-controls-01
```

Artifacts preserve model/data, scope, controls, source hashes, four complete
probe tensor archives per arm, original nominal/stress statuses, measured
probabilities, timing, selection, manifest and result. Master snapshots sources
and records memory growth, TF32, XLA, hardware, environment, charge and git HEAD.

Skeptical audit: the previous forward-statistic rule answered the wrong harm
question. This revision keeps numerical validity but measures an effect on
the actual proposal/MH operation. It does not use high acceptance as mixing
evidence or select on data1704. Neither a pass nor a local failure decides
surrogate-HMC validity; the bounded posterior experiment remains the question.
The weakest assumption is local probe coverage. Downstream convergence and
reference agreement can overturn this selection; healthy stress corners alone
cannot validate behavior between them or throughout the prior.

Attempt01: focused tests found a Path/string serialization mismatch in the new
selection reader (39 passed,1 failed). No GPU job launched. Repaired all saved
path conversions; the full same focused review runs before attempt02. Charge
6.57971 CPU seconds,0 GPU; unchanged sources. This is a harness repair under
the same scientific protocol and allocation.
