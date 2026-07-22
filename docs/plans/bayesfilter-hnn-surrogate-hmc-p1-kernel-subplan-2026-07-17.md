# P1 Subplan: Corrected Neural-Force HMC Kernel

Phase objective: implement the exact batched TensorFlow/XLA kernel proved in
Chapter 48, with full endpoint energy telemetry and fail-closed APIs.

Entry conditions: P0 target/method/default ledgers pass; P1 API and tests are
frozen.

Required artifacts:

- `bayesfilter/inference/neural_force_hmc.py` (or P0-reviewed equivalent);
- typed config/results and frozen force identity;
- batched chain execution, current-value cache, full initial/final momentum,
  endpoint target call, acceptance, and trace telemetry;
- focused unit and XLA tests, including deliberate invalid variants.

Required checks/tests/reviews:

- deterministic involution/reversal and small-dimensional Jacobian tests;
- exact Gaussian moment test and acceptance-to-one as step size shrinks;
- biased position-only force still samples Gaussian moments when mixing is
  adequate;
- negative tests for omitted kinetic energy, momentum-dependent force, direct
  state update, asymmetric early stop, and nonfinite force/target;
- endpoint-value call count and cache test;
- transformed-target fixture proving that endpoint correction includes the
  frozen chart log-Jacobian and fails when it is omitted;
- batch/permutation/replay/shape/dtype tests;
- GPU/XLA canary with memory growth and no NumPy, host callback, or Python
  sample-axis loop in the active path.

Evidence contract: passing P1 establishes implementation agreement with the
proved discrete kernel on tested fixtures. It does not establish useful learned
forces, filtering validity beyond supplied targets, or performance.

Forbidden claims/actions: no general L-HNN implementation as the admitted
route; no posterior-only correction; no state-dependent fallback; no serious
model run; no hiding undefined computation as ordinary rejection.

Exact P2 handoff: all map, energy, Gaussian, negative, batching, and GPU/XLA
checks pass; trace schema supplies the diagnostics required by P2/P3.

Stop conditions: the shared kernel cannot satisfy the Chapter 48 assumptions,
or GPU/XLA is unavailable after trusted probes. Local XLA/serialization bugs
trigger repair up to three attempts.

Phase-end duties: run all focused checks; write P1 result/repair history;
refresh P2; review P2; continue if no real blocker.
