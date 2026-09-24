# Zhao-Cui Algorithm 3 Repair Proposal

Date: 2026-09-11

Status: proposal only; no research-checkout runtime or manuscript source was
modified by this audit continuation.

This proposal follows the source inventory, bounded MathDev checks, and
reconciled CPU evidence at:

- `zhao-cui-integrated-audit-20260911-01/stage-01-source-inventory.md`
- `zhao-cui-integrated-audit-20260911-02/stage-result.md`
- `zhao-cui-integrated-audit-20260911-03/stage-result.md`

## Route classifications

| Candidate action | Classification | Why |
|---|---|---|
| Activate internal TT rank channels in the initializer | `extension_or_invention` at the numerical initializer level | The paper and author source require a fitted TT but do not prescribe this TensorFlow initialization. It repairs capacity/optimization and must be tested as a local implementation choice. |
| Add aggregate finite/normalization guards | `fixed_hmc_adaptation` safety guard | It rejects invalid outputs without changing accepted finite values. It preserves the frozen finite program and is a fail-closed Class B guard. |
| Name the bounded-grid CDF and cell-slope density | `fixed_hmc_adaptation` numerical-law declaration | It makes the implemented sampler’s law explicit. It does not make the law equal to the smooth TT density. |
| Wire a generic consumer through preparation, conditional draw, physical density, weights, and score | `source_faithful` call-chain requirement with local TensorFlow implementation | The endpoint must implement the paper’s Algorithm 2/3 structure and author-source operation; local charts and numerical CDF remain identified adaptations. |
| Differentiate a rebuilt TT, chart, CDF inverse, or state path with respect to parameters | `extension_or_invention` | The current frozen score does not establish this total derivative. No such route should be silently added to the claim-bearing path. |

## Repair 1: rank activation

Target file: `bayesfilter/highdim/squared_tt_engine_v0_tf.py:136-148`.

Current behavior writes nonzero values only at basis index zero, so a
configured rank can begin effectively rank one. The smallest repair should
initialize each internal channel with a deterministic, finite, nonzero basis
component while retaining the boundary cores and declared rank. The exact
perturbation or factorization is a Class C numerical choice and needs a
calibration check; it must not be promoted merely because one fit improves.

Required checks before acceptance:

1. The existing positive rank-two polynomial fixture activates rank two and
   lowers the fit residual to the expected scale without changing the exact
   rank-two warm-start authority.
2. A rank-one configuration remains unchanged.
3. A healthy constant/low-rank fit does not fire a new validity veto.
4. The fit remains finite under the declared condition-number and mass checks.

The repair question is capacity activation, not a claim that one initializer
is globally optimal. Record the chosen perturbation and its failure mode in the
implementation note.

## Repair 2: aggregate fail-closed validity

Target file: `bayesfilter/highdim/zhao_cui_algorithm3_tf.py:171-200`.

Extend the validity mask to include, at every step and at return:

- finite log-sum-exp increment `c`;
- finite cumulative increment sum;
- finite normalized log weights and weights;
- finite strictly positive weight sum within a declared relative tolerance;
- finite positive ESS and finite filtering means; and
- normalized-weight sum close to one under the same precision contract.

The guard must reject the existing extreme finite-input fixture, while a
healthy finite fixture must continue to pass. The guard is observational and
fail-closed; it must not clip, damp, renormalize a nonfinite value, or alter an
accepted finite trajectory. Preserve the rejected result as a negative-control
artifact.

## Repair 3: numerical proposal-law wording

Target manuscript lines: `attempt05_n4_failure_analysis.tex:2788-2790` and
the surrounding proposal-density proposition.

Replace the statement that bounded CDF grids and bisection are “not a new
probability law.” State instead:

- the smooth squared-TT conditional density is the nominal/source-motivated
  object;
- the finite grid, interpolated CDF, and cell-slope density define the actual
  numerical proposal used by the implementation;
- the sampled density must be evaluated using that same cell-slope law; and
- finite bisection residuals are numerical error diagnostics, not a proof of
  exact smooth-TT sampling.

Keep the exact continuous inverse-CDF proposition under its positivity and
continuity assumptions, but label it as a theorem for the nominal law rather
than as a theorem for floating-point grid samples.

## Repair 4: claim-bearing consumer closure

The current bounded name/AST inventory found no non-test consumer. Before any
claim run, trace or add exactly one generic endpoint that:

1. prepares the full adjacent `(current, parameter, previous)` squared TT;
2. performs source-direction mass contractions;
3. builds the upper suffix-conditioned inverse for each fixed particle;
4. evaluates the physical proposal density, including defensive term and
   affine Jacobian;
5. applies the model-to-proposal ratio without an auxiliary ancestor law; and
6. sends the same frozen states and proposal values to the value and analytical
   score evaluator.

The wiring test must fail if the full adjacent TT is discarded before the
conditional density is evaluated, if a reduced diagonal-only path is selected,
or if an ancestor/lookahead probability appears. A test-only import is not
claim-bearing closure.

## Acceptance and stop rules

The four repairs are independent gates. A failed rank candidate blocks rank
promotion but does not reject the Zhao-Cui direction. A failed finite guard or
wrong numerical-law density blocks value/score use until repaired. Missing
consumer closure blocks all end-to-end claims. Do not start GPU, HMC, or
scientific comparison work while any of these gates is open.

The next implementation step, after owner review of this proposal within the
existing authorization, is Repair 2 because it is a no-value-change
fail-closed guard. Repair 1 then requires its explicit Class C calibration.
The manuscript wording and wiring test can proceed as bounded documentation
and test work once their target endpoint is identified.

