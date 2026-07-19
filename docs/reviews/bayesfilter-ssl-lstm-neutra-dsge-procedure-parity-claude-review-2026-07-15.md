# SSL-LSTM NeuTra DSGE-Procedure Parity Claude Review

Date: 2026-07-15

Review mode: bounded, read-only, advisory

Final verdict: `VERDICT: AGREE`

## Review Convergence

The plan and implementation were reviewed iteratively against the frozen
`dsge_hmc` source procedure. Claude was not an execution authority and did not
authorize GPU use, material training, HMC, promotion, or scientific claims.

The plan review first returned `REVISE` for three material gaps:

1. the SSL-LSTM hidden-width transfer was not explicitly tied to the
   dimension-relative Rotemberg/SGU launcher rule;
2. the four-coordinate target chart, order, and fixed translation were not
   bound tightly enough; and
3. local self-consistency tests were proposed where direct comparison with the
   actual sibling `dsge_hmc` classes was required.

The plan was patched to freeze widths `(4,4)`, bind the identity-oriented
coordinate chart and signatures, and require direct explicit-tensor
cross-repository forward, logdet, all-gradient, and Adam-update parity.

The implementation review then returned `REVISE` for four defects:

1. nonfinite gradients were being sanitized instead of failing closed;
2. serialization did not recheck the target and adapter signatures at the
   artifact boundary;
3. a failed restore could partially mutate live trainer state; and
4. legacy schedule/configuration restoration was ambiguous.

The implementation and tests were repaired to reject nonfinite gradients,
rebind signatures during serialization, validate restore payloads before any
mutation, and reject ambiguous legacy state. Focused mutation tests cover each
repair. The final bounded review found no remaining material procedure-parity
defect and returned `VERDICT: AGREE`.

The first bounded result-note review returned `REVISE` for two claim-boundary
issues: its opening could be read as broader paper fidelity, and its 10-hour
request was called a measured minimum without showing the margin over the raw
timing estimate. The result was patched to say local-source parity explicitly
and to identify 10 hours as a prospective contingency cap 19.7% above the
raw 8.357-hour two-seed step estimate.

The second review checked those two repairs on the same exact result path. It
found both fixed, found no new material inconsistency, and returned
`VERDICT: AGREE`.

## Claim Boundary

The verdict supports engineering parity with the cited local `dsge_hmc`
procedure only. It does not establish NeuTra paper fidelity in general,
transport quality on SSL-LSTM, posterior correctness, HMC readiness,
predictive validity, statistical superiority, or default readiness.
