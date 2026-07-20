# SSL-LSTM NeuTra 2,000-Step Adaptive Training Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-neutra-2000-step-adaptive-training-plan-2026-07-20.md`  
Decision: `IMPLEMENTED_FOCUSED_CHECKS_PASSED_NO_MATERIAL_TRAINING_RUN`

## Change

The q-general SSL-LSTM NeuTra runner now uses:

| Setting | Value | Role |
| --- | ---: | --- |
| Final/confirmation maximum | 2,000 optimizer steps | hard cap |
| Validation cadence | 250 steps | heldout check boundary |
| Patience before repair | 250 steps | first no-improvement cycle |
| Learning-rate repair | factor 0.5 | after the first plateau cycle |
| Post-repair no-improvement cycles | 2 | continue at +500, stop at +750 from best |
| Training batch | 480 | unchanged inherited baseline |
| Validation batch | 64 | unchanged inherited baseline |
| Optuna rungs | 50, 100, 200, 400 | unchanged nomination protocol |

The meaningful-improvement definition remains the paired one-sided upper
confidence bound on candidate-minus-best per-sample heldout loss. The runner
restores the best trainer and Adam state before setting the halved learning
rate. A meaningful improvement replaces the best checkpoint and restarts the
plateau clock.

The controller parameter `post_repair_no_improvement_cycles` is explicit and
defaults to `1`, preserving the prior controller behavior for existing callers.
The q-general runner opts into `2`. The parameter is bound into the manifest
and checkpoint hash. A legacy default-policy checkpoint that omits this field
is accepted only by a controller configured for the default one-cycle policy;
it cannot silently resume a two-cycle run.

## Evidence Contract And Interpretation

The focused evidence answers only whether the control protocol, checkpoint
state, and runner contract implement the requested schedule. It does not test
NeuTra transport quality, HMC convergence, posterior correctness, runtime,
optimal training duration, or scientific validity. No material GPU training,
Optuna study, HMC, or default-promotion claim was made.

## Checks

Commands were run from `/home/ubuntu/python/BayesFilter` in the `tfgpu`
environment with GPUs deliberately hidden:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
tests/test_neutra_training_control.py \
tests/test_ssl_lstm_neutra_complexity_training.py
```

Result: `24 passed` with two dependency deprecation warnings.

The suite covers:

- the backward-compatible one-cycle controller sequence;
- the requested `0 -> 250 repair -> 500 continue -> 750 stop` sequence;
- improvement resetting the repair window;
- uninterrupted/resumed controller equivalence;
- minimum-cap and full-repair-window validation;
- legacy checkpoint compatibility and two-cycle mismatch rejection;
- q-general runner constants and contract-smoke payload;
- best-state restoration and LR-setting source paths;
- external pool, sequential Optuna, and material-mode boundary contracts.

Compilation also passed:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
bayesfilter/inference/neutra_training_control.py \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
tests/test_neutra_training_control.py \
tests/test_ssl_lstm_neutra_complexity_training.py
```

The q=20 contract smoke passed with:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
--mode contract-smoke --q 20
```

Output: `JSON_SUMMARY {"mode": "contract-smoke", "q": 20, "status": "PASSED"}`.

`git diff --check` passed for the scoped source, test, plan, and documentation
files. The existing frozen 5,000-step budget JSON was not edited; it remains a
historical receipt for the earlier protocol. The ladder plan and Chapter 28A
now identify this 2,000/250/two-cycle protocol as the subsequent prospective
final-training policy.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Adopt amended controller/runner contract | All focused tests and contract smoke pass | No hard veto | Material loss noise and 2,000-step sufficiency are untested | If separately authorized, run the existing q ladder with fresh artifacts and the revised manifest | No transport, HMC, posterior, runtime, or scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the focused implementation checks |
| Statistically supported ranking | Not applicable; no stochastic candidate comparison |
| Descriptive-only differences | None claimed; historical 5,000-step projections are not reruns |
| Default-readiness | Not established; this is a prospective runner-policy change |
| Next evidence needed | Authorized material GPU training, then downstream transport/HMC gates |

## Post-Run Red Team

The strongest alternative explanation is that 250-step checks are too sparse or
the 2,000-step cap truncates a genuinely improving transport. The current tests
cannot distinguish that because they use deterministic controller fixtures and
do not train. A material heldout history showing continued supported
improvement at step 2,000, or materially worse downstream gates than the prior
protocol, would overturn the practical training-policy choice without
invalidating the controller mechanics.

