# Phase 1 Result

Status: `PASS_GATE_REPAIRED`

All seven known-density/contract fixtures passed in the repaired attempt
`phase1-attempt2` under CPU-hidden TensorFlow 2.20.0/TFP 0.25.0 with XLA. The
affine inverse/Jacobian versus independently parameterized pushed-forward
Gaussian residual was `8.88e-16`; the known mass estimate was exactly `1.75`;
the negative-mode functional error was `0.0111`; the defensive-tail score
second-moment grid integral was `0.9517`; mutation moment errors were `0.0192`
and `0.0330`; and replay metadata residual was zero.

The first wrapper launch failed only because the repository path was not added
before import. That harness repair was applied, tested, and preserved as a
separate attempt history. The fixtures remain finite evidence for their named
quantities, not a q=20 authority proof.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Advance to Phase 2 | all exact/finite fixture gates pass after repair | none | finite fixtures do not prove global assumptions | fresh paired C0/M0 pilot | no q=20 authority admission |
