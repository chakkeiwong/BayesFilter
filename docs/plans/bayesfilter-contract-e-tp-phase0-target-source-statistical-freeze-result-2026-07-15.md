# Contract E--TP Phase 0 Target, Source, And Statistical Freeze Result

metadata_date: 2026-07-15
phase: 0
status: PASS_PHASE0_TARGET_SOURCE_STATISTICAL_FREEZE
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`

## Outcome

Phase 0 passed after repairing one target-labeling defect found during artifact
review. The six primary observed-data rows, deterministic dataset hashes,
parameter coordinates, Zhao--Cui route classifications, source anchors,
comparison roles, seed roles, and statistical nonclaims are frozen in:

`docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json`.

Phase 1 may implement the dense experimental Contract E--TP core. No nonlinear
model chart may be prepared until its model-specific parameter region is
reviewed. SIR is the exception: its existing reviewed `[-0.5,0.5]^3` log-scale
region is retained.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Freeze exactly six primary observed-data rows | Pass: registry and seven semantic tests agree | No scoped SIR component promoted | Future client models are outside this registry | Implement dense core against the 2D witness | Not all repository models validated |
| Preserve Zhao--Cui comparator boundaries | Pass: LGSSM oracle alias and predator retained-grid substitutions are forbidden | SIR component/full-score boundary is explicit | Several comparators require later certification or implementation | Phase 6 comparator work after core/model validation | Not Zhao--Cui equivalence |
| Freeze uncertainty without inventing an equivalence threshold | Pass: 16-replicate pilot and 16/32/64 precision ladder recorded; all row margins remain unavailable | `0.05*sqrt(p)` is FD-only | Model-specific cross-method margins remain scientifically open | Report descriptive intervals until a margin is derived | No reasonable-match/equivalence claim |
| Freeze source support | Pass: local Zhao--Cui JMLR PDF/text and pinned author-code manifest are hashed | No paper claim used as derivative correctness | Live retraction/forward-citation metadata was not queried | Inspect additional primary sources only when later mathematics requires them | Not a complete literature review |

## Target Registry

| Row | Target observation shape | Parameter dimension | Zhao--Cui status |
| --- | ---: | ---: | --- |
| LGSSM | `50 x 3` | 5 | Real fixed-TT route missing; current oracle adapter forbidden as Zhao--Cui evidence |
| Actual SV | `1000 x 1` | 2 | Scalar fixed-TT adapter exists; recertification required |
| KSC-SV | `1000 x 1` | 2 | Fixed-TT extension exists; recertification required |
| Generalized SV | `1008 x 1` | 3 | Fixed-design extension exists; recertification required |
| Predator--prey | `20 x 2` | 6 | Source model exists; retained-grid route forbidden; source-route comparator required |
| Austria SIR | `20 x 9`, latent `d=18` | 3 | Fixed-TTSIRT component route exists; full observed-data total score remains blocked |

The JSON preserves exact serialized-tensor SHA-256 values rather than copying
them into this prose result.

## SIR Boundary

The registry correctly records the existing SIR d=18 fixed variant. P90/P91
component/value-bridge evidence is mandatory regression evidence, not a missing
implementation. The full filtering-gradient comparison remains gated by:

```text
BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED
BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED
```

The local complete-data component score cannot be substituted for the marginal
observed-data score.

## Repair Record

The first generated registry incorrectly placed generalized SV's
`log(y^2+1e-6)` flow observations in the `target_observations` field. That
quantity is a Gaussianized LEDH proposal surface, while the target correction
uses raw observations. The builder was repaired to store raw observations as
the target and the transformed values under `proposal_flow_observations`. A new
test requires distinct hashes and the correct roles. This was a target-identity
repair, not metadata cleanup.

## Checks Actually Run

CPU-only choice: `CUDA_VISIBLE_DEVICES=-1` was set before TensorFlow import. The
TensorFlow process printed CUDA plugin-registration and `cuInit` messages even
with devices hidden; these are startup diagnostics and not GPU evidence.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m compileall -q \
  docs/benchmarks/build_contract_e_tp_phase0_registry.py \
  tests/highdim/test_contract_e_tp_phase0_registry.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/build_contract_e_tp_phase0_registry.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp pytest -q \
  tests/highdim/test_contract_e_tp_phase0_registry.py
python -m json.tool \
  docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json
git diff --check -- <Phase 0 paths>
```

Results:

- registry generation passed;
- JSON parse passed;
- focused tests: `7 passed`;
- compilation passed;
- diff hygiene passed.

`jq` was unavailable for an optional display-only summary. This did not affect
the registry or its semantic tests.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | Dirty pre-existing research worktree; unrelated changes preserved |
| Environment | Current TensorFlow environment; CPU-only registry generation |
| Data | Deterministic repository generators, seeds `81100`, `81101`, `81103`, `81104`, `81105` |
| Role seeds | Preparation `82100:82115`; validation `82200:82215`; audit `82300:82315` |
| Wall time | Registry generation approximately 6.7 seconds; focused tests 0.04 seconds |
| Artifacts | Builder, registry JSON, seven semantic tests, local Zhao--Cui PDF/text, this result |

## Handoff

Phase 0 gate: `PASS`.

Phase 1 must implement only the dense TensorFlow projection primitives and the
documented 2D LGSSM witness. It must not wire Contract E--TP into canonical
Contract E, current model runners, the leaderboard, or HMC.
