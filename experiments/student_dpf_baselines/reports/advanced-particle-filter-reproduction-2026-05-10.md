# Advanced particle filter reproduction report

## Date

2026-05-10

## Scope

This report records Phase G1 of
`docs/plans/bayesfilter-student-dpf-baseline-gap-closure-plan-2026-05-09.md`.
It covers the vendored `advanced_particle_filter` snapshot only.

Snapshot commit:
`d2a797c330e11befacbb736b5c86b8d03eb4a389`.

## Environment

Observed command environment:

```text
Working directory: /home/ubuntu/python/BayesFilter
PYTHONPATH: experiments/student_dpf_baselines/vendor
Python: 3.13.13
Python executable: /home/ubuntu/anaconda3/envs/tfgpu/bin/python
NumPy: 2.1.3
SciPy: 1.17.1
TensorFlow: 2.20.0
TensorFlow Probability: 0.25.0
TensorFlow devices: CPU
```

TensorFlow reported no CUDA-capable device.  The tests below were therefore
CPU runs.

## Commands and outcomes

| Command | Status | Interpretation |
| --- | --- | --- |
| `env PYTHONPATH=experiments/student_dpf_baselines/vendor python experiments/student_dpf_baselines/vendor/advanced_particle_filter/tests/test_basic.py` | passed | Direct README-style integration test passed: 3 passed, 0 failed. |
| `env PYTHONPATH=experiments/student_dpf_baselines/vendor pytest experiments/student_dpf_baselines/vendor/advanced_particle_filter/tests/test_filters.py -q` | passed | Main NumPy filter pytest suite passed: 22 passed. |
| `env PYTHONPATH=experiments/student_dpf_baselines/vendor pytest experiments/student_dpf_baselines/vendor/advanced_particle_filter/tests/test_kernel_pff.py -q` | partial failure / terminated | The run emitted `.F.` and then continued for several minutes without completing.  The process was terminated after the reproduction gate was already satisfied by the previous two commands. |

## Selected stdout summaries

`test_basic.py` reported:

```text
Results: 3 passed, 0 failed
KF  - Mean RMSE: 0.1480, Log-lik: -25.68
EKF - Mean RMSE: 0.1480, Log-lik: -25.68
UKF - Mean RMSE: 0.1480, Log-lik: -25.68
BPF - Mean RMSE: 0.1476, Avg ESS: 84.2
EDH Flow - Mean RMSE: 0.1478, Avg ESS: 500.0
EDH PFPF - Mean RMSE: 0.1643, Avg ESS: 44.7
LEDH Flow - Mean RMSE: 0.1473, Avg ESS: 500.0
LEDH PFPF - Mean RMSE: 0.1571, Avg ESS: 60.8
```

`test_filters.py` reported:

```text
22 passed in 12.84s
```

`test_kernel_pff.py` emitted:

```text
.F.
```

and did not complete before termination.

## Decision

Status: `adapter_justified`.

Rationale:

- `advanced_particle_filter` now has a completed targeted reproduction in the
  local snapshot and environment.
- The direct integration script and main NumPy filter pytest suite both passed.
- The kernel PFF partial failure must remain visible, but it does not veto
  minimal adapter work because the initial adapters target Kalman/bootstrap-PF
  style linear-Gaussian smoke fixtures, not kernel PFF.

## Caveats

- Passing student tests is not a BayesFilter correctness certificate.
- TensorFlow-side HMC/DPF tests were not used as the Phase G1 gate.
- Kernel PFF behavior requires a later focused reproduction/debug pass before
  being used as comparison evidence.
