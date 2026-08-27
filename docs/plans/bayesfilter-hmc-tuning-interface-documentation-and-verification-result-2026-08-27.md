# BayesFilter HMC Tuning Interface Documentation And Verification Result

Date: 2026-08-28

Status: `IMPLEMENTATION_AND_TERMINAL_REVIEW_PASS_CLOSEOUT_PENDING`

Implementation commit:
`8b201a55a6cd453ca199f3e75755f1ea4bf5489e`.

Plan:
`docs/plans/bayesfilter-hmc-tuning-interface-documentation-and-verification-plan-2026-08-27.md`.

## Outcome

BayesFilter now has one executable capability registry and two active public
artifact-authority HMC tuners:

1. `tune_hmc_kernel` for an ordinary adapter-owned exact target and score; and
2. `tune_fixed_transport_hmc_kernel` for one identity-bound frozen nonlinear
   transport with the Jacobian-corrected transformed target and matching score.

The direct neural-force chain runner remains diagnostic mechanics. A new
repository-issued `HMCTuningRunnerBinding` lets its raw-coordinate,
position-only force enter `tune_hmc_kernel` without gaining artifact authority.
The binding is reused across mass adaptation, epsilon and leapfrog-count
selection, screening, fresh verification, and repair. It rejects a bare
callable, target or coordinate mismatch, missing telemetry or identity, and the
runner's direct identity-mass fallback.

The ordinary default verifier now consumes its existing rank-normalized split
and folded split R-hat result at the final handoff boundary. A failed or
inconsistent R-hat result cannot issue a final kernel. The existing threshold
is `1.01`; no threshold was introduced or retuned by this work. Ordinary tuning
ESS admission remains disabled, and neither tuning R-hat nor acceptance proves
retained posterior convergence.

The Markdown guide, monograph chapter, generated route tables, executable
examples, concise `AGENTS.md` rule, contradiction ledger, and downstream
migration guidance are committed with the implementation. MacroFinance and
dsge_hmc were inspected but not edited. In particular, the dsge_hmc backend
lock was not changed.

## Skeptical Audit Disposition

The pre-execution audit checked the baseline, proxy promotion, stop conditions,
coordinate and target assumptions, environment, and whether each planned
command could answer its stated question. No intervening commit between the
reviewed baseline and implementation start changed the three tuner source
files. Separately, the plan limited execution to CPU-hidden interface tests;
no GPU or sampler campaign was needed to answer the documentation contract.

The audit's two material implementation questions were answered as follows:

- The typed neural-force design is feasible without creating a third public
  artifact-authority tuner. Fake-runner call-ledger tests show that the bound
  runner reaches every relevant stage and receives stage-selected mass,
  epsilon, and leapfrog count.
- The prior ordinary final-admission behavior was wrong relative to the stated
  R-hat handoff policy. Focused tests reproduce and close that gap. ESS remains
  explicitly outside ordinary tuning admission rather than being silently
  invented as a new gate.

During the focused matrix, an existing windowed-mass compatibility test exposed
a deterministic contradiction: a non-authoritative exception payload was
required to retain `error_type`, but the source hard-coded it to `None`. The
repair retains only `type(exc).__name__`; arbitrary messages and exception
details remain redacted, and operational handoff authority is unchanged.
The checked anchors are
`tests/test_hmc_kernel_tuning_windowed_mass.py:1684` and
`bayesfilter/inference/hmc_kernel_tuning.py:9161`.

## Verification Evidence

All TensorFlow commands below used `CUDA_VISIBLE_DEVICES=-1` before import.
They are interface and documentation checks, not GPU or sampler evidence.

| Check | Result |
| --- | --- |
| Route inventory and stale-entry check | pass; no unclassified or stale discovered tuning routes |
| Generated Markdown/LaTeX renderer `--check` | pass; no drift |
| Capability and documentation contracts | 16 passed |
| Ordinary verification, handoff, and runner subset | 54 passed, 197 deselected by the planned `-k 'verification or handoff or runner'` expression |
| Fixed-transport tuning and binding files | 47 passed |
| Plan-specified neural-force subset | 3 passed, 18 deselected by the planned `-k 'tuning or runner or coordinate or artifact'` expression |
| Additional complete neural-force file | 21 passed |
| Route-selection example | pass; selected both public tuners and rejected the low-level runner as a tuner |
| Ordinary smoke example | pass; returned the permitted contract-only `hard_veto` status under its deliberately tiny budget |
| Fixed-transport smoke example | pass; returned the permitted contract-only `no_viable_candidate` status under its deliberately tiny budget |
| Python compilation of new modules, renderer, examples, and contract test | pass |
| Full `latexmk` build | pass; 525-page PDF |
| Undefined-reference, undefined-citation, and duplicate-label log scan | pass; no matches |
| Rendered chapter inspection | pass; registry columns and source listings are readable without overlap |
| `git diff --check` and staged diff check | pass |

An independent fresh agent was given only
`docs/reference/hmc-tuning-interface.md` and three target descriptions. It
selected `tune_hmc_kernel` for ordinary coordinates,
`tune_fixed_transport_hmc_kernel` for a genuine frozen transport, and a typed
neural-force binding passed to `tune_hmc_kernel` for an arbitrary raw
position-only force. It also rejected direct
`run_full_chain_neural_force_hmc` use as full tuning. Result: `CHECK: PASS`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain the guide, registry, binding, and R-hat repair | Section 16 route, behavior, example, and build gates pass; terminal review ends in `VERDICT: AGREE` | No interface or document contradiction remains in the focused matrix | The typed binding is tested with deterministic fixtures, not a downstream BGS campaign | Migrate each downstream consumer under its own pin and tests | Posterior convergence, sampler superiority, target validity, GPU/XLA readiness, performance, production readiness |
| Keep exactly two public artifact-authority tuners | Registry validation and signature tests pass | Low-level runners and historical/diagnostic routes remain non-authoritative | A future algorithm may need a distinct target contract | Add a route only after its own reviewed capability and end-to-end admission tests | That every future force or transport fits an existing route |
| Do not update the dsge_hmc lock | Downstream compatibility preconditions are not met | TensorFlow-only BGS tuning remains a downstream hard blocker | The eventual BayesFilter implementation selected for that policy is undecided | Revise and review the dsge_hmc integration plan before any lock selection | BGS HMC readiness or scientific validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No remaining interface/document hard veto was observed in the focused BayesFilter matrix. Downstream migration and repository closeout limitations below remain open. |
| Statistically supported ranking | `N/A`; no stochastic method ranking was performed. |
| Descriptive-only differences | Test counts, runtimes, build size, and consumer symbol counts are descriptive only. |
| Default readiness | Not established by documentation or interface correctness. |
| Next evidence needed | Downstream pin/schema tests; a separately reviewed TensorFlow-only decision for dsge_hmc; retained-chain diagnostics for any later sampler claim. |

## Environment And Provenance

| Field | Value |
| --- | --- |
| Implementation-start commit | `1a284ec2d09b7776b7e44fecd211e9f8e7a3ade3` |
| Implementation commit | `8b201a55a6cd453ca199f3e75755f1ea4bf5489e` |
| Python | `3.13.13` |
| TensorFlow | `2.20.0` |
| TensorFlow Probability | `0.25.0` |
| Device policy | deliberate CPU-only checks; `CUDA_VISIBLE_DEVICES=-1`; TensorFlow reported no physical GPU |
| Package/environment mutation | none |
| Serious sampler or research run | none |
| Random seeds | example/config fixture seeds only; no stochastic scientific inference |
| Existing primary artifacts | implementation commit, this result note, generated route tables, contradiction ledger, downstream guidance |
| Terminal review artifact | `docs/plans/bayesfilter-hmc-tuning-interface-documentation-terminal-claude-audit-result-2026-08-28.md`; final substantive verdict `AGREE` |

### Exact commands and wall times

The focused command matrix was:

```bash
python scripts/inventory_hmc_tuning_routes.py --check
python scripts/render_hmc_tuning_interface_docs.py --check
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_public_api.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_hmc_kernel_tuning_windowed_mass.py -k 'verification or handoff or runner'
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_fixed_transport_hmc_tuning.py tests/test_fixed_transport_hmc_binding.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_neural_force_hmc.py -k 'tuning or runner or coordinate or artifact'
CUDA_VISIBLE_DEVICES=-1 python docs/examples/hmc_tuning_route_selection.py
CUDA_VISIBLE_DEVICES=-1 python docs/examples/hmc_tuning_ordinary.py
CUDA_VISIBLE_DEVICES=-1 python docs/examples/hmc_tuning_fixed_transport.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
rg -n 'Undefined references|Citation .* undefined|multiply defined' main.log
git diff --check
git ls-files --others --exclude-standard
```

The capability/documentation pytest command took 5.6 seconds wall time; the
ordinary filtered command 13.5 seconds; fixed transport 13.3 seconds; the
neural filtered command 5.0 seconds; and the additional full neural file 10.4
seconds. The inventory, renderer, and three examples completed as one
concurrent group in 4.6 seconds. The final LaTeX rebuild took 5.8 seconds.
These are descriptive local timings, not performance comparisons. The
repo-wide untracked-file command returned the unrelated exception recorded
below rather than satisfying its planned empty-output gate.

## Changed Surface

The implementation commit changes 25 files. The main ownership groups are:

- capability and binding code in `tuning_contract.py`,
  `neural_force_hmc.py`, `hmc_kernel_tuning.py`, and package exports;
- behavioral tests in the ordinary and neural-force suites plus the new
  documentation contract;
- the HMC guide, monograph chapter, executable examples, generated route
  tables, and renderer;
- the concise agent rule, contradiction ledger, reviewed plan disposition, and
  downstream migration guidance; and
- one pre-existing duplicate LaTeX label rename needed to make the required
  repository-wide duplicate-label log gate exact.

No benchmark, retained chain, posterior result, downstream source, backend
lock, package, or environment was changed.

## Remaining Closeout Limitations

1. The first trusted `git fetch origin` attempt failed because `github.com`
   could not be resolved. A later retry succeeded; `origin/main` remained at
   `1a284ec2d09b7776b7e44fecd211e9f8e7a3ade3`, and
   `git merge origin/main` reported `Already up to date`. The closeout commit
   and push remain to be performed.
2. `git ls-files --others --exclude-standard` is not repo-wide empty because
   `docs/plans/bayesfilter-ssl-lstm-q20-phase52-governance-migration-2026-08-28.md`
   is concurrent, unrelated, claim-supporting Phase 52 work. It must be tracked
   by that work's owner, not ignored or folded into this implementation commit.
   Every untracked file created by this HMC task is included in the
   implementation commit.
3. MacroFinance and dsge_hmc migration status is guidance only. Neither
   downstream repository has run the new compatibility tests or selected this
   commit.

## Post-Run Red Team

The strongest alternative explanation for the clean route matrix is that the
registry and tests agree with each other while both omit a real consumer
behavior. A downstream contract test against the exact commit, especially the
BGS arbitrary-force path, would overturn the claim that the interface is
sufficient for that consumer. This is the weakest part of the current evidence.

The strongest alternative explanation for the typed binding result is that the
deterministic fake runner exercises call plumbing but not a difficult target's
numerical behavior. A stage that silently switches runner, coordinates, mass,
epsilon, or leapfrog count in a real integration would overturn the implemented
full-stage claim. The existing per-stage ledger is necessary but does not
establish sampler quality.

The strongest alternative explanation for the R-hat repair is that a fixture
proves serialization and classification but not the reliability of R-hat for a
particular posterior. A result artifact issuing a final kernel after a failed,
missing, inconsistent, or cap-exhausted verifier would overturn the interface
claim. Passing the gate still would not prove retained posterior convergence.

The strongest alternative explanation for documentation usability is that a
fresh agent can answer three prepared cases but a user can still encounter an
unrepresented target contract. A downstream agent selecting a chain runner as
a tuner after reading the pinned guide would overturn the usability finding.
Rendered readability and compilation do not certify a human voice or complete
coverage of future algorithms.

## Nonclaims

This result does not establish posterior convergence, target correctness,
sampler superiority, statistical ranking, effective-sample-size adequacy,
performance, GPU/XLA readiness, production readiness, scientific validity, or
downstream compatibility. The smoke examples' finite completion does not
promote their deliberately under-budgeted candidates. The interface work
repairs ownership, identity, admission, and documentation behavior only.
