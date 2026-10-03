# Reset memo: q=20 performance and whitening continuation closeout

Date: 2026-09-02  
Authority: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Closed plan: `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`  
Status: `M3C_CLOSED_NO_NOMINATION_N3_BLOCKED`

The M3-C continuation completed its bounded N0--N2 work. The valid GPU
receipt is `n0-n2-02-gpu/run_manifest.json` (manifest hash
`a73661f50e2b559e54549c950ccaadaa0b027c1e9dbbdecd74971f692e68f4e4`), and
the fresh CPU N1 control is `n1-cpu-05/run_manifest.json` (manifest hash
`649b10aaa9a00f4d5df8645c2f3abca89b24001861afc2425a0a52c51193d78c`).

The first GPU attempt is preserved but is invalid for cross-arm interpretation
because it assigned a different held-out validation bank to each arm. The
harness was repaired to share a validation seed per seed index before the
second run. That repaired run is the sole N2 comparison evidence.

N1 confirms that the proposed fast grouped TFP transition is not equivalent to
the scalar transition (state error `1.9390212244446556`, target error
`0.86641743413635`, gradient error `1.9390212244446556`, and log-acceptance
error `0.008664174341362962`). The explicit scalar row-loop control is exact.
The fast path therefore remains unintegrated.

N2 produced 9/9 finite candidates with 12/12 valid updates, but no arm met the
predeclared two-of-three-seed, 10%-score-RMS nomination rule. This is a
candidate-screen failure, not a rejection of the entire research direction.
No default, HMC route, whitening claim, or Phase 9B entry was changed.

The route-specific regression passed `38 passed`. The broader five-file focused
suite returned `113 passed, 2 failed`: one known ordinary-tuner migration
failure and one missing private LGSSM retained-samples fixture. Those failures
are outside M3-C and remain explicitly recorded repository debt; they are not
evidence that the whole repository is green.

The next action is intentionally not automatic. Further optimization or a new
grouped-kernel design needs a new reviewed plan and budget. M3 replay remains
terminal; old partial calls cannot be resumed or pooled into a new result.
