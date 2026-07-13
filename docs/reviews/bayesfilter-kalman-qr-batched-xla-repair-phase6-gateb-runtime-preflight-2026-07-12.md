# Phase 6 Gate B Runtime Preflight

Date: 2026-07-12

Status: `SKEPTICAL_RUNTIME_AUDIT_PASS_EXACT_GATE_B_COMMAND_AUTHORIZED`

## Authority Chain

| Artifact | SHA-256 / identity |
| --- | --- |
| Parent Phase 6 plan | `b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b` |
| Opening hash ledger | `9261e0c560ede29dc6893e0ffe3769cd762b38f3dd651af6dfcfa2f90dce1911` |
| Gate B proposal | `e1b4cabba3dfd1ca292c4d7842d02ba86273001275b5f0a3b69ed0851a0ec823` |
| Gate B authority ID | `ff9913a2bb8ad101fb9e4edab1a021aeff627228830c125b1296dbeaf874e837` |
| Claude proposal review record | `166bd5594d01371e6b08186f885e1cc3f06130d0072aa4a5363802412dc6e0e8` |
| Detached attestation | `583e7842c3af2ebe0e00598224a86dd5cbf9c2627f0a22ca0638e25f870cd153` |
| Review strength | `claude_opus_max` |

`validate_phase6_runtime_authority(...)` passed all ten checks:
closed schema, schema identity, gate identity, authority identity, proposal
digest, plan digest, review digest, exact agreeing verdict, review strength, and
timestamp.

Protected algorithm hashes remain:

- `bayesfilter/linear/kalman_qr_tf.py`:
  `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`;
- `bayesfilter/linear/kalman_qr_derivatives_tf.py`:
  `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`;
- `bayesfilter/linear/qr_factor_tf.py`:
  `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`.

## Final Skeptical Audit

- Wrong baseline: absent. The proposal binds the reviewed historical failure
  class and exact Phase 4/5 evidence, not a convenient weak comparator.
- Proxy promotion: absent. Trace census is a structural prerequisite only;
  pilot runtime is a viability screen only. Neither can establish method
  superiority, production readiness, or scientific validity.
- Missing stop conditions: absent. Authority/evidence/provenance/process
  ambiguity, common structural invalidity, corrupt/oversized artifacts, unsafe
  cleanup, and budget exhaustion have exact closure rules.
- Unfair comparison: absent. Both primary methods use matched identities,
  deterministic fixtures, fresh sequential children, one requested CPU thread,
  the same v4 measurement contract, and method-local outcomes.
- Hidden assumptions: closed. The proposal binds one command, 36 trace and two
  pilot schedules, source/runtime/config/fixture/schedule fingerprints, empty
  runtime inputs, CPU/GPU-hidden environment, 60-second execution, 70-second
  lifecycle, one shared 160-second paired-cell cap, 3000-second TERM deadline,
  45-second KILL grace, and 3045-second monotonic authority.
- Stale context: absent. Proposal, review, attestation, plan, opening ledger,
  protected sources, CLI surface, and no-worker/no-target-output state were
  rechecked immediately before this record.
- Environment mismatch: explicit. This is a reviewed CPU-only target diagnostic
  with `CUDA_VISIBLE_DEVICES=-1`, CPU device, one requested thread, and XLA only
  for the pilot. It is not the repository's default GPU production route.
- Artifact fitness: direct. Lossless bounded GraphDefs, transition ledgers,
  child bytes, process evidence, dependency manifests, and strict evaluators
  answer the gate; timing summaries are explanatory only.

Audit result: `PASS`.

## Evidence Contract

The exact question, comparator, criteria, vetoes, explanatory diagnostics,
nonclaims, and artifact paths are those in the reviewed Gate B subplan. In
particular, pilot launch requires the full conjunctive trace predicate:
`trace_common_valid=true`, all six cohort comparisons pass, and every
`rejected_differences` list is empty.

Valid candidate failure rejects only that candidate under the cap. It does not
reject the harness, target, implementation, research direction, or later
reviewed GPU repair unless a declared common/continuation veto fires.

## Authorized Action

Only the exact Gate B command recorded in the reviewed subplan and immutable
proposal is authorized. Gate C, GPU, HLO, comparison, source edits, timeout/cap
changes, and scientific/default/product claims remain forbidden.
