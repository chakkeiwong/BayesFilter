# SSL-LSTM q=20 NeuTra global-mixing execution-plan review

Date: 2026-08-19

Reviewer: Claude Code worker, launched through the repository-approved bounded
worker wrapper under trusted execution.

Reviewed path:

`docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md`

Prompt boundary: read-only review of exactly the path above; no edits, commands,
agents, or repository-wide review. The question was whether the plan was
scientifically and operationally ready for execution under the repository
policy.

Verdict: `REVISE`

## Material findings

1. The scientific endpoint named only whole-path tests. It did not freeze the
   test statistic, calibration pass/fail rule, or consequence of calibration
   failure.
2. Phase caps and permitted retries could consume the full campaign budget
   without leaving an explicit preflight or repair allocation.
3. Setting `CUDA_VISIBLE_DEVICES=1` while also passing `--device 1` left physical
   versus post-mask logical device semantics ambiguous.
4. Numeric provenance was missing for the 600-row batch, the 128-transition
   canary, and the predictive sample size, horizons, and alpha.
5. The plan required diagnostics over four parameters without explicitly
   asserting that the target dimension is four and that this is complete
   coverage.
6. The statement that `central-07` remained unevaluated conflicted with the
   inherited canary source, which would evaluate that bank if launched.
7. The HMC rule said to try the second candidate only if the first failed but
   also specified a branch for both candidates passing; that branch was
   unreachable and the candidate order was not fully frozen.

## Residual cautions

- Rank-normalized diagnostics on a binary mode indicator require careful
  tie/degeneracy handling; direct per-chain transitions should remain the
  principal anti-pooling evidence.
- Replay-weight degeneracy is explanatory unless a predeclared numerical
  invariant fails.
- Timeout termination can leave partial files, so claim-bearing writes should
  be fresh-root and atomic.
- The initial-state perturbation mapping and HMC tuning-helper signature should
  be verified before launch.
- `q=20` must not be confused with the four-dimensional inferred parameter
  vector.
- Five separate 1% tests do not create a joint equivalence decision; even under
  independence the corresponding familywise false-rejection arithmetic is
  approximately 4.9%.

## Disposition

The execution plan was revised before any GPU experiment. The revised plan
freezes the repository whole-path energy/permutation statistic and decision
rule, makes true-vs-true a mechanics calibration whose realized p-value is
explanatory, reconciles aggregate budgets, defines physical/logical GPU
semantics, records numeric provenance, asserts full four-parameter coverage,
reserves `central-07` in the new campaign, and makes seed 3 a fallback attempted
only after seed 2 fails. Source repairs and focused tests remain prelaunch
requirements.
