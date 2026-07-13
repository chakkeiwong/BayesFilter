# Phase 4 Result: XLA Value/Score Gate

Date: 2026-07-09

Status: `PASSED`

## Scope

Phase 4 checked whether the deterministic multidimensional LGSSM target
value/score path can compile and execute under TensorFlow XLA with
`jit_compile=True`. It did not run HMC, tune a kernel, train NeuTra, generate
posterior samples, or make posterior recovery claims.

## Skeptical Pre-Run Audit

| Risk | Audit Finding |
| --- | --- |
| Wrong baseline | The gate used the Phase 3 `T=120` fixture and Phase 2 config hash. |
| Proxy promotion | Compile success is used only as an XLA admissibility gate, not as HMC readiness. |
| Missing stop condition | Nonfinite value/score, retracing, JIT fallback, or runtime autodiff would veto. |
| Environment mismatch | Command set `CUDA_VISIBLE_DEVICES=-1`; XLA initialized for `Host`. |
| Artifact mismatch | The artifact records value, score, timing, HLO size/hash, JIT flags, and nonclaims. |

Verdict: `PASS_TO_EXECUTE_PHASE4`.

## Implementation Artifacts

- Driver:
  `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
- Focused test:
  `tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- XLA gate artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json`

## Artifact Summary

| Field | Value |
| --- | --- |
| Schema | `bayesfilter.deterministic_lgssm_hmc_tuning_xla_score_gate.v1` |
| Artifact hash | `sha256:f945e98ad2aac75cb2998e51855a717e9b9894384f254c29861e4911d35593a9` |
| Config hash | `sha256:683e45cef9a46e14a3ee2de3e51d5fc19a0512feb43e376e30c2da19e1a2ccb0` |
| Fixture hash | `sha256:346d2932ac329a477b35530010cf2dff6d4cf2022f003216b1a46a19bbca54ac` |
| `jit_compile` | `true` |
| `jit_compile_false_runtime_executed` | `false` |
| Runtime autodiff tape executed | `false` |
| Value finite | `true` |
| Score finite | `true` |
| Score shape | `[18]` |
| Concrete function count | `1` |
| Compile plus first execute seconds | `7.762520305113867` |
| Warm execute seconds | `0.10597397992387414` |
| HLO metadata available | `true` |
| HLO byte count | `3427939` |
| HLO hash | `5000f5bde5c2851a19eebfa3e7d682a49cd9e6e9502481ee54ffe38b8335756f` |
| Vetoes | `[]` |

## Checks Run

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `5 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage xla_score
```

Result: XLA gate JSON written. TensorFlow emitted CUDA plugin/cuInit warnings
despite `CUDA_VISIBLE_DEVICES=-1`; XLA service initialized for `Host` and
TensorFlow logged that a cluster was compiled using XLA. This is recorded as
CPU-hidden XLA evidence, not GPU evidence.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json
```

Result: JSON parse check passed.

```text
git diff --check -- docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py tests/test_deterministic_lgssm_hmc_tuning_driver.py docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json
```

Result: passed.

## Evidence Contract Assessment

| Field | Assessment |
| --- | --- |
| Question | Can the LGSSM target path compile and evaluate under XLA-only execution? |
| Primary criterion | Met: the value/score path executed with `jit_compile=True`, finite value, finite score, and one concrete function. |
| Veto diagnostics | No nonfinite value/score, no JIT-disabled run, no runtime autodiff tape, and no retrace veto. |
| Explanatory diagnostics | Compile/execute timing, warm execute timing, HLO byte count, HLO hash, value, and score. |
| Not concluded | No HMC tuning success, convergence, posterior recovery, sampler superiority, default readiness, production readiness, or GPU claim. |

## Decision Table

| Field | Decision |
| --- | --- |
| Phase decision | `PASS_TO_PHASE5` |
| Primary criterion status | XLA value/score gate passed |
| Veto diagnostic status | No Phase 4 veto triggered |
| Main uncertainty | Geometry/mass initialization has not yet been run against this target |
| Next justified action | Execute Phase 5 deterministic geometry and mass initialization |
| What is not concluded | No HMC run, no burn-in, no retained sampling, no convergence, no posterior recovery |

## Plain-Language Gate

Claimed target: compile and evaluate the LGSSM log-probability and score path
with XLA/JIT enabled.

Computed quantity: one finite scalar log probability and one finite 18-vector
score at the configured truth/prior-mean probe point, plus bounded XLA timing
and HLO metadata.

Verdict: `correct` for Phase 4 XLA compile admissibility under the named
CPU-hidden command; `not checked` for HMC tuning, posterior recovery, NeuTra,
GPU execution, and scientific validity.
