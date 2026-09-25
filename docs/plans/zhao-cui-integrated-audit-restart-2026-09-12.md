# Zhao-Cui integrated audit continuation brief

Stage 01 source inventory was refreshed on 2026-09-12 at HEAD `47176bce7cdaa91ddd4466c39f90a11ad005c800`; see `docs/plans/artifacts/zhao-cui-integrated-audit-20260911-continue-01/stage-01-source-inventory.md`. Existing runtime edits remain untouched.

Checked: Zhao-Cui paper Algorithm 2/3 anchors, author MATLAB `full_sol`, `marginalise`, `eval_cirt_reference`, and driver hashes; local preparation/evaluator/compiler; manuscript Algorithm 3 labels; AST call sites. MathDevMCP independently proved only the scalar weight cancellation under `q != 0`, with boundaries recorded in `logs/mathdev-weight-cancellation.json`.

Current decision: `REVISE`; rank activation, aggregate fail-closed checks, numerical-CDF law consistency, and claim-bearing consumer wiring remain open.

Next exact task: write the bounded no-code Repair 2 decision note defining aggregate validity invariants and two deterministic regression fixtures. Preserve the separate rank Class C calibration as a proposal. No runtime edits, GPU, training, HMC, numerical comparison, or claim-bearing evaluation.
