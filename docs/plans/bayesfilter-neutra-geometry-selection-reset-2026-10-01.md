# Resume state after geometry and selection repair

The owner requested another code/math audit and repair after the preceding
gap-closure cycle completed. That new cycle has now also completed, with an
important negative finding: **neither mixture seed has adequate evidence of
rare-region sampling under the repaired validation requirements**.

Read `bayesfilter-neutra-geometry-selection-results-2026-10-01.md` and its plan.
Evidence root is the existing campaign's `geometry-selection-20261001-r1/`.
No worker is active and no numerical phase is queued. Do not replay completed
HMC attempts or reuse the final reference to select another checkpoint.

Checked facts:

- All nine saved maps passed manual-score/autodiff/finite-difference/roundtrip
  checks on the recorded inputs. Large residuals are real at those inputs.
- KL does not control score residuals without regularity; the result note gives
  a counterexample. Exact mixture transports themselves can have huge
  derivatives near a valley, so large Jacobian alone is not a defect.
- Refinement now preserves an eligible checkpoint shortlist, each map with its
  own1000-point probe. Qualification tests those maps downstream, then opens
  final reference once. Forward KL no longer supplies the sole HMC map.
- The executed v3 posterior profile checked18 quantities. Seed11 failed all
  four maps/twelve tested members. Seed37 RKL1024 at epsilon.0625/L18 formally
  passed v3 after7000 warmup and6000 retained transitions per chain.
- Terminal inspection found only2 valley visits in24000 retained draws for
  that formal pass, both in one chain. Its rare-event estimate is.00008333
  versus exact.00134990. The v3 additive reference-SD allowance was too large
  to test this probability. This is a validation-design defect, not a derivative
  defect or proof of an asymptotically wrong HMC target.
- The prospective v4 profile requires both event outcomes per retained chain
  and removes the additive SD allowance for binary probability agreement.
  Saved seed37 draws fail both checks. Exact-iid controls passed53/64 runs at
  2000 draws/chain and63/64 at4000; these do not calibrate dependent HMC.

The original numerical execution is frozen under `source/`. The prospective
v4 validation repair is under `source-v4/` and was exercised only in CPU exact
reference and retrospective checks. Never relabel v3 outputs as v4 confirmation.
The completed controller remains frozen at v3; a future v4 numerical run needs
a fresh output root and source snapshot under a new reviewed subplan.

Engineering checks:48 distinct tests passed. Terminal artifact audit checked
631 frozen source files,109 input hashes,18 complete standard probes and17
member archives without integrity errors. A saved numerical veto remains a veto
on controller resume. Original failed candidates and reference artifacts remain.

Resource charge:1209.986 GPU process-seconds and1778.424 CPU core-seconds,
including CPU reference controls and terminal inspection. Remaining shared
allocation:81480.811 GPU process-seconds and170544.791 conservative CPU
core-seconds. No new allocation, package install, commit, merge or push.
The shared worktree has extensive unrelated changes; preserve them.

Next justified experiment: a bounded local HMC transition/energy diagnostic
from physical valley and ordinary posterior starts, with a fixed saved map
and step-size hypotheses justified by measured local curvature. It must use
the public fixed-transport tuner for any retained qualification. The existing
observations suggest unresolved localized geometry but do not prove that
smaller steps, more training, a new objective or a new architecture will solve
it. Do not launch a new full campaign merely to chase a passing result.
