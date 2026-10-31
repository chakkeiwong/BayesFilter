# Current replication root-cause investigation

Owner request: trace the current code and investigate remaining replication gaps.
This investigation continues the authorized replication campaign. It does not
change either author's algorithm or the accepted qualification criteria.

## Question and evidence contract

Why do the TensorFlow controlled-SMC fitting path and the TensorFlow-trained
KDM checkpoints still differ from their pinned author baselines? Trace each
claim-bearing consumer to the executed numerical implementation and distinguish
demonstrated causes, untested hypotheses, and missing qualification evidence.

Comparators are the preserved author MATLAB-language implementation under the
recorded Octave/LAPACK environment, and Younis/Sudderth source commit
`b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18` under its recorded PyTorch environment.
Neither comparator establishes equality to the authors' original hardware runs.

The investigation passes when each remaining gap has current code/source anchors,
the strongest available discriminating evidence, and a bounded next action.
Any new deterministic diagnostic must use matched inputs and preserve its full
results. A numerical mismatch is a repair trigger; invalid inputs, changed
baseline, missing provenance, nonfinite controls, or exhausted budget stop that
diagnostic. Existing failed screens remain failed. Aggregate audit counts and
successful training completion are engineering evidence, not numerical parity.

Training loss, local gradient error, condition number, source variability,
runtime and trace counts are explanatory. They cannot establish no-worse final
filter performance. No population ranking, qualification, new default, HMC or
robustness claim follows from this investigation. Heuristic competition is not
the question: these are source-translation checks, not an algorithm promotion.

## Work and assumption audit

1. Trace current dispatchers, stage initialization/transfer, objective,
   resampling/gradient estimator, optimizer, stopping and checkpoint selection.
   Recheck relevant paper equations and actual pinned author code.
2. Trace controlled-SMC polynomial design, normal equations, conditional solver,
   rank decisions, and forward-policy consumption. Inspect saved failing systems
   and the previous solver intervention, including its limitations.
3. Check whether Python, NumPy, retracing or XLA claims match the actual hot path.
   Separate reference/orchestration code from numerical kernels.
4. Use the smallest discriminating checks for any newly found mismatch. Prefer
   saved inputs/checkpoints to additional full training. Inspect optimizer state
   history and matched stochastic inputs before attributing accumulated training
   quality to floating-point arithmetic.
5. Record established causes, open hypotheses and remaining scope in a result
   note; refresh the master and concise checkpoint.

Source configurations, dtype, particle count, data and stopping rules remain
baseline choices with source provenance. Independent native random streams are
valid distributional choices but do not isolate numerical drift; matched-input
tests are required for that attribution. Float64 is a diagnostic reference, not
a substitute for the float32 target. Source intrinsic failures are permitted
baseline outcomes; extra port failures remain disqualifying. A better conditioned
solver is a different floating-point implementation and cannot silently replace
the author baseline.

## Execution and budget

Read-only inspection plus at most 1,800 CPU seconds and 1,800 GPU seconds of
focused diagnostics, charged to the existing allocation (starting remaining
243,038.982123 CPU seconds and 159,978.197543 GPU seconds). No complete training
or qualification launch. Use fresh directories under
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-cause-20261008-*`.
Record actual commands, hashes, seeds, timing and device policy there. GPU work
uses the approved tftwogpu interpreter with escalation and memory growth; CPU
reference commands hide CUDA before imports. No environment/package changes.

Skeptical pre-execution audit: the comparison must target translation fidelity,
not paper superiority; the pinned Octave baseline is narrower than MATLAB
reproduction; independent training seeds confound a numerical-drift attribution;
single-step agreement cannot prove accumulated-update parity. The ordered checks
above address these risks and do not tune acceptance thresholds on failures.
Local self-review passes this bounded investigation scope. Runtime test commands
will be recorded before launch once the relevant call-chain gap is identified.

### First discriminating diagnostic: optimizer arithmetic

Code inspection found that TF computes Adam bias corrections in the parameter
dtype, while the installed author's CPU PyTorch Adam uses host double scalars
on its non-capturable path. Feed identical saved gradients into the actual TF
update method and source Adam, bypassing clipping in both arms to isolate Adam.
Use the TF-trained MDPF best checkpoint with zero moments or saved mature moments
(step 24,332), and the same frozen fixture learning rates. These are optimizer
mechanics probes, not continued training.

Predeclared endpoints: updates 1, 2 and 16. Compare GPU/XLA float32 and float64,
CPU graph float32 reference, and a diagnostic GPU float32 intervention computing
only the two bias-correction scalars in float64. Runtime code stays unchanged.
Every input gradient is identical; each graph traces once. The float64 primitive
control must have maximum absolute error below 1e-10. Float32 differences are
descriptive; they cannot attribute training quality. Preserve both method bodies.

Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B
docs/benchmarks/diagnose_kdm_optimizer_localization.py --output
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-cause-20261008-01

Run with escalation; source child hides CUDA, candidate configures memory growth.
Record child commands, environments and elapsed time. No untouched qualification
data is consumed. Self-review passes: this isolates one scalar-arithmetic choice;
remaining differences may include learning-rate rounding and fused update order.

First diagnostic completed on the system-default RTX 4080 SUPER. Preserve it as
mechanics evidence. Repeat once in root-cause-20261008-02 with the original
training manifest's RTX 5080 UUID explicitly pinned; this corrects an execution
identity omission without changing inputs, method or test criteria. The original
diagnostic source is archived in the first output directory.

## Completed investigation and terminal review

Investigation completed on 8 October 2026. See
[the root-cause result](controlled-smc-kdm-root-cause-result-20261008.md) and
`artifacts/controlled-smc-kdm-replication-20260928/root-cause-20261008-summary.json`.
Both diagnostic executions completed with finite updates, passing float64
controls, one trace per graph and identical comparison rows on the two GPUs.
The original unpinned run remains preserved; the retry pins the original
training GPU UUID. No runtime numerical behavior was changed.

Self-review checked the consumer call chains against the source and relevant
paper equations, the arithmetic intervention, controls, input/source/result
hashes, local result links and executable Python syntax. The key limitation is
unchanged: the isolated Adam difference is not an attribution of the trained
KDM quality gap. Conditional gradient checks do not establish distributional
sampling parity or coupled training behavior. Exact solver coefficients are
not the downstream acceptance criterion. The result keeps these distinctions
and the previous failed screens explicit.

Charges: 7.697021533 CPU seconds for numerical references, 28.441637605 GPU
seconds for complete candidate processes, plus a separately declared 120 CPU
seconds for routine validation/inspection. Remaining active allocation after
charging once: 242911.285102 CPU seconds (67.48 hours) and 159949.755905 GPU
seconds (44.43 hours). No campaign job is running.

Next stage: specify the bounded coupled KDM update diagnostic, preserving
optimizer/controller state, row order and underlying random variates. Localize
the first divergence before any new full training run; continue cSMC fitting
and failed-policy checks under their own bounded plans. Both replications
remain incomplete, with accepted numerical criteria unchanged.
