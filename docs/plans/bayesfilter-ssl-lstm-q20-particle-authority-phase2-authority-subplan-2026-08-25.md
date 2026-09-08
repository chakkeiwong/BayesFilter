# Phase 2 Fresh q=20 Authority Pilot Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_GATE_CANDIDATE`  
Budget cap: `14400 s`  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase2`

## Objective

Generate fresh q=20 particles and establish whether C0 and M0 can produce an
auditable, target-linked authority candidate. The historical six-bank replay
may be measured as context but must not seed, replace, or certify this run.

## Entry gate

Phase 1 must pass the affine identity, frozen-protocol hash, known-density mass
fixture, and support/metadata checks. If only an auxiliary fixture fails, run
the declared repair or continue with the unaffected C0 diagnostic while keeping
M0 blocked. Do not silently lower the M0 gate.

## Protocol

1. Use calibration partitions to choose a tempering schedule, mutation
   controls, defensive-mixture epsilon, and particle ladder.
2. Freeze and repository-hash every claim-run stage, trigger, mutation control,
   proposal-law version, target signature, and seed domain before drawing.
3. Run C0 and M0 on identical target partitions and paired seeds. C0 remains a
   normalized descriptive comparator; M0 records unnormalized mass and all
   Feynman-Kac metadata.
4. Use fresh `N=100`, `N=300`, and `N=600` arms only as a hypothesis ladder. The
   old `N=100` and pooled `N=600` artifacts are context, not warm starts.
5. Record target value/status, proposal/transition/observation/mutation log
   densities, ancestry, mode labels, seeds, worker identity, protocol hash,
   and geometry signatures for every retained row.
6. Keep an online-adaptive variant separate and label it C0 descriptive until
   its actual conditional law is independently tested.

## Gates and roles

| Diagnostic | Role | Condition |
|---|---|---|
| finite/status and metadata | hard veto | every retained row valid and auditable |
| support/defensive denominator | promotion veto | positive on declared target support |
| frozen-law hash | hard veto | exact pre/post parity |
| M0 mass and mode stability | promotion criterion | independent seeds/partitions with declared uncertainty; no rank from descriptive-only short runs |
| mutation invariance | promotion veto | tractable bridge check remains valid at executed controls |
| mode occupancy and ESS | explanatory/repair trigger | report tails and coverage; cannot certify discovery |

## Failure interpretation

Missing negative mode with valid bookkeeping is candidate evidence and triggers
fresh proposal/mutation repair, not a research-direction veto. Invalid density,
support, or protocol identity blocks M0 and all claims that depend on it. A
resource failure gets a bounded repair under the same scope.

## Required artifacts

- fresh C0/M0 run manifests and per-seed receipts;
- protocol/hash and proposal-law files;
- mass/mode/ancestry diagnostic table with uncertainty status;
- phase2 result and repair/refresh note;
- refreshed Phase 3 controls and partitions.

## Executed pilot receipts

Two fresh CPU/XLA pilots passed the hard bookkeeping gates:

- `phase2-attempt1-n16`: both C0 and M0, `43.2 s`;
- `phase2-attempt2-n100`: both C0 and M0, `146.7 s`.

The N=100 M0 receipt is the Phase 3 input. Identity mutation is recorded as an
exact invariant reference kernel with no mixing guarantee. Mode occupancy and
root counts remain descriptive; M0 is not yet an admitted authority.
