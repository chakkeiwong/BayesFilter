# SIR-UKF NeuTra Owner Exclusion Reset Memo

Date: 2026-07-31

Status: `TERMINAL_OWNER_EXCLUSION`

## Owner Decision

UKF does not work for SIR. `SIR-UKF` is removed from active NeuTra testing.
It must not be selected by the all-model master, included in future NeuTra
campaigns, repaired, retuned, trained, or advanced to HMC without a new owner
direction.

## Preserved Evidence

The implementation and completed artifacts remain as historical diagnostic
evidence. They are not active candidates and do not authorize reentry. The
terminal target-specific screen is preserved at:

```text
docs/plans/artifacts/bayesfilter-sir-ukf-neutra-target-specific-20260730/
  screen-attempt-01/SIR-UKF/result.json
```

Its decision was `NO_SURVIVING_RECIPE`: all four recipes and all three seeds
were hard-vetoed by invalid exact-target status. No final training or HMC was
launched.

## Active Registry State

`SIR-UKF` is absent from `EXECUTABLE_CELLS` and appears only under
`OWNER_EXCLUDED_CELLS` with state
`OWNER_EXCLUDED_METHOD_NOT_APPLICABLE`. The remaining active master registry
contains five executable cells, six ordinary blocked cells, and one
owner-excluded historical cell.

Collected SIR-UKF target/parity and high-dimensional UKF-on-SIR scout tests
were removed. The focused parity driver was removed. The historical mixed P6
SIR target-design driver now fails immediately before TensorFlow import or
artifact creation. Historical P6/P7 artifacts and audit readers remain
preserved and must not be interpreted as active testing.

The final active-path audit also found two routes outside the NeuTra registry:

- `docs/benchmarks/run_fixed_sir_sgqf_validation.py` optionally evaluated a
  SIR-UKF comparator. That arm was removed. The active comparator option now
  evaluates bootstrap PF only, records the owner exclusion, and emits schema
  `bayesfilter.fixed_sir_sgqf_validation.v2`.
- `scripts/p76_bounded_ukf_minibatch_pilot.py` constructed a spatial SIR UKF
  scout. Its dependent P76 corrected-metric diagnostic and P77 training driver
  reached the same context. All three are now fail-closed historical
  tombstones that exit before TensorFlow import or artifact creation. Their
  prior result documents remain historical provenance.

Reusable UKF implementation symbols and initializer mechanics remain in the
repository for provenance and unrelated models. They are not registered or
executed as SIR tests. Reentry of any UKF-on-SIR route requires a new owner
direction.

## Verification Contract

The exclusion is complete when:

- the active registry contains five executable, six blocked, and one
  owner-excluded cell;
- every all-model action rejects `SIR-UKF` before TensorFlow import;
- the SIR-SGQF validation driver has no SIR-UKF callable; and
- each retired P76/P77 entry point fails from an empty working directory before
  TensorFlow import and creates no requested output.

Historical P0/P7 registry builders and audit readers are intentionally outside
this execution contract because they only reconstruct or verify preserved
evidence.

## Verification Results

The following explicit CPU-only contract suite passed with GPU devices hidden:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 MPLCONFIGDIR=/tmp/bayesfilter-mpl \
python -m pytest -q \
  tests/test_neutra_all_models_end_to_end_contract.py \
  tests/test_sir_filter_neutra_target_design.py \
  tests/highdim/test_fixed_sir_sgqf_tf.py \
  tests/test_multimodel_neutra_p7_audit.py

66 passed, 2 dependency deprecation warnings
```

`python -m py_compile` passed for the three retired entry points and the active
SIR-SGQF validation driver. `git diff --check` passed. The terminal Python
source search found only:

- explicit owner-exclusion guards and tombstones;
- historical P0/P7 registry and audit readers; and
- preserved, non-registered implementation symbols.

No active test or benchmark driver imports or invokes a SIR-UKF computation.

## Nonclaims

This owner decision removes the UKF-on-SIR testing direction. It does not alter
the completed PP-UKF result, reject SIR with other filtering methods, or make a
claim about UKF on unrelated models.
