# HMC Tuning Guide and Policy Repair Result

Date: 2026-09-01  
Controlling plan: `docs/plans/bayesfilter-hmc-tuning-guide-policy-repair-plan-2026-09-01.md`  
Status: `PASS_REPAIR_COMPLETE_WITH_SCOPED_RESIDUALS`

## Scope and decision

This repair addressed the mismatch exposed by the q=20 Phase 9A preflight:
the guide described finite-length HMC resonance, while the executable
fixed-transport tuner still used a one-dimensional acceptance ladder and tests
that assumed monotone acceptance. The repaired claim-bearing route is now
`measured_joint_grid_v1`. It measures every declared `(step_size, L)` pair,
uses replicated fixed-kernel efficiency evidence, and performs a disjoint
held-out check. The former directional ladder remains available only as
`legacy_directional_diagnostic_v1` and cannot issue a handoff.

This result establishes guide/implementation/test consistency for the tuning
interface. It does not establish posterior convergence, mode discovery,
sampler superiority, HMC readiness for q=20, or high-dimensional scaling.
Phase 9B remains closed until a fresh q=20 campaign satisfies its own
target-specific evidence contract.

## Repairs made

- `docs/reference/hmc-tuning-interface.md` and
  `docs/chapters/ch21b_hmc_tuning_interfaces.tex` now define mean Metropolis
  probability separately from binary acceptance, explain fixed-L resonance,
  and state that an acceptance target is an efficiency heuristic.
- `FixedTransportHMCKernelTuningConfig` exposes an explicit bounded step-size
  grid and the `measured_joint_grid_v1` policy. It rejects underspecified,
  over-cap, or over-budget grids before a chain starts.
- The measured route evaluates all declared pairs and records attempted and
  measured-pair status. Selection uses replicated ESS-per-gradient evidence and
  a disjoint held-out fixed-kernel verification.
- Missing/non-finite telemetry and non-positive retained movement remain hard
  ineligibility conditions. A finite acceptance value outside the target band
  is a descriptive repair trigger on the measured route, not a validity veto.
- Candidate, result, kernel, and selection payloads now expose mechanics,
  tuning, authority, and posterior-status fields; schemas were bumped so old
  artifacts cannot be silently upgraded.
- Initial-to-first-retained displacement is no longer counted as retained
  fixed-kernel movement.
- The active q=20 preflight and HNN caller use explicit measured grids. The
  July LGSSM campaign is explicitly quarantined as legacy diagnostic-only,
  rather than silently issuing an old one-L handoff.

## Verification record

Run manifest:

| Field | Value |
| --- | --- |
| Git revision | `54201f5cd925ed15036bad8156606b812d53b045` (worktree also contains unrelated pre-existing changes) |
| Python | `3.13.13` |
| TensorFlow / TFP | `2.20.0` / `0.25.0` |
| Device mode | CPU-hidden (`CUDA_VISIBLE_DEVICES=-1`); no GPU or HMC campaign launched |
| Allocator setting | `TF_FORCE_GPU_ALLOW_GROWTH=true` (harmless CPU-hidden launch provenance) |
| JIT | Test fixtures exercise both the declared TensorFlow paths and explicit non-XLA diagnostic fixtures; no production claim is made |
| Seeds | Fixed fixture/config seed ledgers in the tested modules; no new stochastic model campaign |
| Persistent experiment output | None; pytest temporary directories only |

Commands and outcomes:

| Command | Outcome | Evidence role |
| --- | --- | --- |
| `bash /home/ubuntu/python/BayesFilter/scripts/run_hmc_tuning_policy_tests.sh` | Pass, 18 tests | Policy, documentation, compile wrapper |
| `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 python -m pytest -q tests/test_hmc_tuning_policy_repair.py tests/test_fixed_transport_hmc_tuning.py tests/test_fixed_transport_hmc_step_cap.py` | Pass, 58 tests | Focused measured-grid, legacy-boundary, movement, cap, and handoff checks |
| `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 python -m pytest -q tests/test_hmc_tuning_posterior_oracle.py -k fixed_transport` | Pass, 3 tests; 7 deselected | Analytic Gaussian transport and held-out failure behavior |
| `python -m py_compile ...` (tuner, callers, and policy tests) | Pass | Import/compile correctness |
| `python scripts/render_hmc_tuning_interface_docs.py --check` | Pass (also run by wrapper) | Generated route-table consistency |
| `git diff --check` | Pass | Patch whitespace/integrity check |

The full historical LGSSM test file was also attempted. Nine tests passed and
one pre-existing static-input test failed because the archived comparator file
`docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/phase7_campaign/private/retained_samples.npz`
is absent. This is an artifact-availability failure outside this repair; it is
not evidence against the repaired tuner or the HMC research direction.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Guide and executable policy agree | Markdown, LaTeX, registry, and tests name the same measured joint policy | Pass | None found in the bounded scope | Keep this contract as the active interface | No target-specific tuning quality |
| False high-acceptance tuning is blocked | Harmonic phase and adversarial fixtures prevent acceptance-only promotion | Pass | Short fixtures do not model q=20 geometry | Run the declared q=20 grid under a fresh campaign | No posterior correctness |
| Legacy compatibility is contained | Legacy route is explicitly diagnostic and handoff builder rejects it | Pass | Historical callers outside the inspected set may still need migration | Use the source scan before activating a caller | No claim about old artifacts |
| Default readiness | Requires target-specific held-out and retained-chain evidence | Not assessed | q=20 fresh grid, mode coverage, and long-chain diagnostics | Keep Phase 9B closed and prepare a new subplan | No default/HMC admission |

## Inference status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | Passed for the repaired policy fixtures and focused route tests | No finite/telemetry/schema regression was found in scope |
| Statistically supported ranking | None | Unit fixtures and two-replication screens do not support ranking stochastic candidates |
| Descriptive-only differences | Acceptance, ESJD, ESS-per-gradient, runtime, and warning counts in fixtures | Useful for diagnosis, not superiority claims |
| Default-readiness | Not established | A real q=20 target-specific campaign is still required |
| Next evidence needed | Fresh measured `(epsilon, L)` q=20 grid, disjoint heldout, retained-chain R-hat/ESS, mode/target checks | Must use a new versioned output root and the repaired policy |

## Failure classification and red-team review

The first selective oracle failure was a stale test that attempted to make
held-out acceptance out-of-band and expected a hard failure. Under the repaired
contract that acceptance is descriptive; the fixture was changed to inject a
non-finite held-out target value, which is the intended hard failure. The
LGSSM missing-file failure is an artifact problem. Neither failure invalidates
the target, transport mathematics, or the research direction.

An independent ordinary-tuner oracle test was also run as a regression check
and remains red: the ordinary route returned `budget_exhausted` with
`verification_rhat_above_threshold_or_cap_hit` and no hard veto, while the
legacy test still asserts `calibration.passed is True`. The ordinary tuner
module was not changed by this repair, so this is outside the fixed-transport
guide scope and is not evidence against `measured_joint_grid_v1`. It needs a
separate ordinary-tuner fixture/verification plan; this result deliberately
does not relax the R-hat or movement criteria to make that assertion pass.

The strongest alternative explanation is that the policy fixtures are too
small to expose a caller-specific integration bug. The result that would
overturn this conclusion is a fresh target-specific run showing that a caller
can still select an unmeasured pair, conflate acceptance semantics, or issue a
legacy handoff. Such a result triggers a new localized repair, not relaxation
of the guide.

The bounded source audit found two active artifact-authority tuner routes in
the registry. Older benchmark scripts and discovery/refinement helpers remain
historical or diagnostic and therefore are intentionally not migrated into the
claim-bearing route by this repair; invoking one of those callers must fail
closed or remain explicitly diagnostic.

## Approval and allow-list boundary

The user request was sufficient authorization for this trusted local repair.
No network, package installation, external message, destructive Git action,
GPU launch, or per-command click was needed. The only narrow optional host rule
is:

`bash /home/ubuntu/python/BayesFilter/scripts/run_hmc_tuning_policy_tests.sh`

The relative form is equivalent from the repository root. The separate future
q=20 GPU rule, if authorized later, should match exactly:

`bash /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`

and expose physical GPU 0; it must not broaden permission to arbitrary shell
or Python commands.

Do not add broad `bash`, `python`, `codex`, package-manager, network, or
arbitrary-GPU entries. A later GPU q=20 campaign needs one separate trusted
approval for its bounded launcher, with memory growth configured and verified
before TensorFlow import. That approval is outside this result.
