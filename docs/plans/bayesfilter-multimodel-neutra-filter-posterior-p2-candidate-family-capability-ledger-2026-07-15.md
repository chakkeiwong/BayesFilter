# P2 Candidate-Family Capability Ledger

Date: 2026-07-15

| Family | Status | Evidence | Consequence |
| --- | --- | --- | --- |
| Plain dense IAF | `AVAILABLE` | `bayesfilter/inference/neutra_training.py` and frozen loader tests | May be screened only after target/comparator admission |
| Affine transport | `BASELINE_ONLY_NOT_ENHANCED_LEARNED_FAMILY` | Frozen affine loader and LGSSM historical baseline | Cannot satisfy the second learned candidate arm |
| Second enhanced learned family | `UNAVAILABLE_CAPABILITY_NOT_EXECUTED` | Repository search found no trainer/loader for spline, Brenier/ICNN, RealNVP, or another distinct family | Recipe failure cannot become cell rejection; adding a family is a later scientific-method expansion |

This capability gap does not invalidate target/filter admission and is not a
reason to invent an alias for a wider/deeper dense IAF. It becomes binding only
if the plain dense-IAF arm fails and the program would otherwise claim all
predeclared candidate families were tested.
