# LGSSM/SIR cost diagnosis, 2026-10-11

The active question is why the five-model master is slow, whether the actual
force/endpoint call chain is XLA compiled, and whether Python loops explain
the cost. This is a bounded engineering diagnostic within the master's cost
assessment. It does not launch HMC, change the algorithm, retune controls, or
establish posterior validity.

The source audit finds serial TensorFlow loops over chains and analytical
score directions. A four-chain, one-leapfrog transition needs 32 full LGSSM
filter calls (p=4) or 24 SIR calls (p=3). The filter uses all-pairs Gaussian
mixture densities and finite Sinkhorn transport at every observation. At
N=4032 the transport policy gives four 2016-by-2016 tiles; the tiled source
recomputes cost/tangent tiles within every row/column update. SIR has d=18,
observation dimension 9; LGSSM has d=3 and observation dimension 3. These are
source-level operation counts, not a measured kernel-time breakdown.

The comparator is the unchanged current implementation, with the controls
selected in preparation attempt 06. Measure the first call separately from
two warmed calls of the actual scalar force and independent endpoint. Also
time the existing map construction, proposal draw, mixture-density, and reset
functions at the first observation and first parameter direction, using the
same data/design. Component timings explain cost; they are not additive exact
shares of a fused full filter. All numerical kernels have explicit stable
signatures and jit_compile=True. Record graph loop/callback counts and the
XLA must-compile attribute. Fetch computed scalar outputs to synchronize
timing without copying particle arrays to the host.

Evidence contract and intent ledger:

- Pass criterion: preserved exact-scope controls/data, finite valid outputs,
  compiled calls and separate first/warm timings in durable JSON.
- Expected mechanism: repeated primal work across score directions/chains,
  quadratic particle-pair work, and larger SIR matrices. Compilation and
  independent FP64 endpoints are additional hypotheses, not assumed causes.
- Continuation veto: device/memory policy failure, stale selection, unexpected
  nonfinite output, or the diagnostic wall deadline. Preserve partial results.
- Repair trigger: harness error; at most one focused repair within the same
  total allocation. Timing alone cannot promote or reject a scientific method.
- Explanatory diagnostics: component timing, graph nodes, allocator peak,
  call counts, fixed-input replay, and first-minus-warm elapsed time. The last
  quantity includes tracing and first-use execution effects, not pure compile.
- No conclusion about convergence, speed superiority between methods,
  posterior correctness, or a reliable hours-to-completion forecast.

Budget: at most 360 wall seconds for two sequential GPU worker processes;
no more than two warmed repetitions per timed function. Use the idle RTX 4080
SUPER by UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b. The RTX 5080 used in
attempt 07 currently has another workload. This hardware difference forbids
direct subtraction of these timings from attempt 07 or treating them as the
5080's sampling rate. Each worker configures and verifies memory growth before
any tensor allocation, uses ordinary FP32/TF32 off in the filter, and retains
the existing FP64 independent endpoint as a recorded comparison exception.

Default/assumption audit: N4032 and FP32/TF32 off are owner choices. T50,
parameterization, data/design seeds and selected controls come from the active
master and are held fixed. The first observation/direction and two repetitions
are convenience choices for localization, not representative posterior
benchmarks; full-horizon scalar-force timings check whether the localized
cost is consequential. Existing exact Kalman and bootstrap endpoints provide
simple target-cost sanity checks. No new numerical protection or approximation
is introduced. GPU architecture and first-use compilation are explicit risks.

Skeptical audit: passed for diagnosis only. The old pilot includes compilation,
SIR's timeout is not a numerical failure, controls differ from the earlier
readiness run, and an isolated component loses enclosing-graph optimizations.
The diagnostic preserves those distinctions and cannot nominate a new method
or change a default. The full force is called through the master's binding,
so finding a fast unused helper cannot satisfy the call-chain check.

Command (the driver saves full worker logs and stops at its total deadline):

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/diagnose_surrogate_hmc_five_model_cost.py --output docs/plans/artifacts/surrogate-hmc-five-model-cost-20261011-01 --gpu-uuid GPU-68251639-fe82-8f81-3ccc-2953c32e805b --budget-seconds 360
```

Result: this file's companion
`surrogate-hmc-five-model-cost-result-2026-10-11.md`, with full structured
evidence under the command's versioned output directory.
