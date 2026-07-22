# P1 Skeptical Pre-Execution Audit

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `PASS_AFTER_PLAN_REPAIR`

## Audit Decision

P1 may execute only after three planning defects were repaired in the P1
subplan. The admitted runtime scope is a complete repository-owned analytic
Gaussian canary plus negative tests against the eleven blocked registry rows.
No model-cell target, HMC run, training run, or target signature is admitted.

## Findings And Repairs

| Audit risk | Finding | Repair |
| --- | --- | --- |
| Wrong baseline | The draft inherited “P0-frozen targets”, but P0 issued zero posterior signatures. | Replace the entry condition with attempt-04 evidence and restrict positive runtime evidence to a synthetic complete posterior. |
| Proxy promotion | Canary loss/acceptance could be mistaken for model evidence. | Classify all canary numerical metrics as explanatory; primary pass is harness integrity and fail-closed behavior. |
| Hidden identity assumption | The existing SSM target signature binds mathematical metadata but not execution dtype or inspected batch dependency closure. | Issue a campaign-level typed identity binding contract signature, adapter binding, dtype, recomposition, and scope. |
| Circular recomposition | Reusing the production final posterior assembler would let the same omission pass twice. | Require separate prior, likelihood, and Jacobian callables and reject the adapter final assembler as a component. |
| State inflation | A recipe failure could be recorded as a cell rejection or a blocked cell could advance. | Enforce explicit transition edges and separate `RECIPE_REJECTED` from `CELL_CANDIDATE_REJECTED`; reject every transition from a P0-blocked row in P1. |
| Environment mismatch | CPU tests can accidentally initialize a GPU; GPU evidence can run without memory growth. | Hide CUDA for CPU checks; require trusted GPU, XLA, GPU placement, and verified memory growth for the canary. |
| Unanswered command | The draft had no exact P1 commands/output root. | Freeze the commands and fresh phase-p1 attempt root in the refreshed subplan. |
| Misleading success | A successful canary could be reported as a nonlinear-model result. | Preserve explicit nonclaims and require the result to report all eleven model cells still `TARGET_BLOCKED`. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Analytic Gaussian canary with exponential chart | P1 interface-test need | Complete exact posterior with independently visible prior, likelihood, and nonzero Jacobian terms | Too easy numerically | Identity/recomposition and mutation tests, including omitted Jacobian, are primary | Test fixture only |
| `float64` | Existing NeuTra/HMC shared modules | Matches current target/trainer/HMC implementation | GPU throughput is not representative | Record dtype and make no performance claim | Existing execution invariant |
| Dense IAF, tiny topology | Existing trainer | Exercises real frozen-artifact path | Could be mistaken for a recipe default | Explicit synthetic-only nonclaim | Interface fixture |
| 64 steps, batch 64 | Bounded canary budget | Enough to compile and execute the full graph | Does not assess transport quality | No downstream quality promotion | Convenience canary choice |
| Four chains, 16 draws | Shared HMC minimum and tiny health check | Exercises a single batched XLA chain invocation | Cannot establish convergence | No R-hat/ESS/posterior claim | Smoke only |
| Stateless domain-separated CPU seeds | Repository policy | Worker scheduling must not change sample identity | Partition collision/order drift | Worker-count invariance test | Shared execution invariant |

## Evidence Contract

Question: can the shared harness preserve complete target identity, independent
recomposition, batching, state, artifact, archive, and GPU/XLA execution
boundaries while refusing incomplete model cells?

Primary pass: focused tests and trusted canary pass; every P0 cell remains
blocked; all negative substitutions and circular recomposition attempts fail.

Vetoes: caller-stamped identity, incomplete target issuance, circular
recomposition, cross-target artifact, forbidden state transition, scalar or
callback target, unseparated archives, no GPU/XLA/memory growth, or missing
manifest fields.

Not concluded: any declared model/filter cell works; any training recipe is
adequate; HMC converged; a filter posterior is correct; or NeuTra is ready for
production/scientific promotion.

## Pass Rationale

After repair, the command artifacts directly answer the shared-harness question,
promotion and continuation vetoes are distinct, the environment matches the
CPU/GPU policy, and no model evidence can be inferred from the canary. P1 may
execute within its existing 16 CPU-hour and 2 GPU-hour budget.
