# P6 SIR-SGQF HMC Attempt 01 Repair Record

Date: 2026-07-16

Attempt root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/plain-hmc-affine/attempt-01`

Classification: `PLANNING_AND_TUNING_SELECTION_DEFECT`.

The six short probes were finite and health-valid, but the harness selected
step `0.40` by maximum minimum bulk ESS while its modern R-hat was `2.7322`.
This violated the active BayesFilter policy that fixed-kernel tuning admission
uses the maximum of rank-normalized split and folded rank-normalized split
R-hat at `<=1.01`. Acceptance `0.9961` was explanatory and could not repair the
failed tuning gate.

The invalid selection then produced three archived warm-up chunks of 1,000
draws per chain. Read-only diagnostics on their recent windows gave maximum
modern R-hat `1.2122`, `1.2285`, and `1.6761`, all above the warm-up threshold
`1.05`. The attempt was interrupted before chunk 4 to avoid spending more of
the comparator budget on a kernel that had never passed tuning admission.

No target, identity, geometry, filter, HMC implementation, or GPU failure was
observed. Attempt-1 probe rows remain valid tuning-only evidence and are bound
by `tuning_selection.json` SHA-256
`76e204264d38a51079be8866a39b01e038ffc86030666371b748b89bd3b0a5be`.
The attempt-1 selected kernel and every attempt-1 sample are diagnostic only;
they must not be pooled into the retry or used for comparator admission.

Repair: retain the frozen step grid, order health-valid probes by lowest
short-probe modern R-hat and then bulk ESS, then verify candidates with
disjoint 1,000 burn-in plus 1,000 draws.
Only a finite, health-valid candidate with modern R-hat `<=1.01` may enter the
fresh comparator warm-up/retained controller. A focused regression requires
high acceptance and high short-probe ESS to remain unable to bypass modern
R-hat.

Attempt cost: approximately 37 minutes of probes plus 20 minutes for three
warm-up chunks. Remaining work stays within the existing six-GPU-hour
SIR-SGQF comparator bucket.
