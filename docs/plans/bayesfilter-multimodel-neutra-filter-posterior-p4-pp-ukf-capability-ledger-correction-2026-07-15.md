# P4 PP-UKF Capability-Ledger Correction

Date: 2026-07-15

Successful immutable target-admission attempt:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/pf-target-admission/attempt-03-20260715T121908Z/`.

The typed PP-UKF target identity and `POSTERIOR_IDENTITY_ADMITTED` transition
are valid. Their target, execution, status, recomposition, registry, and artifact
hash bindings do not depend on the training-family list.

The attempt-local `cell_ledger.json` used `CampaignCellLedger`'s historical
default `("plain_dense_iaf", "enhanced")`. That default conflicts with the P4
capability audit: only `plain_dense_iaf` exists in current BayesFilter code;
`enhanced` is `UNAVAILABLE_CAPABILITY_NOT_EXECUTED`.

Active P4 accounting is therefore:

| Family | Status | Execution permission |
| --- | --- | --- |
| `plain_dense_iaf` | available baseline | may be screened only after comparator admission |
| enhanced family | `UNAVAILABLE_CAPABILITY_NOT_EXECUTED` | must not be fabricated or budgeted as executable |

The successful attempt remains immutable. Future P4 ledger construction now
passes `required_candidate_families=("plain_dense_iaf",)` explicitly. This
correction does not issue, alter, or promote a transport, recipe, HMC result, or
scientific claim. Failure of the only available family cannot establish a
cell-wide NeuTra rejection.
