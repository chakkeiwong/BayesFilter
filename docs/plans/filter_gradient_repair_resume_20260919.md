# Filter and gradient repair recovery

Current checkpoint through 04004, September 26: the native KDM
repair and the shared LEDH safety/stage repair are implemented on branch
`repair/filter-gradient-xla-validation-20260918`. The KDM repair is pushed at
`66df4de03`; the safety/stage changes are staged in the current worktree and
have not yet been committed.

The safety helper now uses direct TensorFlow broadcasting for validity masks and
rejects nonfinite Cholesky factors while preserving squeezed shapes. The LGSSM
flow and multi-step analytical stage recurrences use native `tf.while_loop` and
the existing native determinant authority. CPU/GPU FP64 and FP32 records,
analytical finite differences, changed inputs, exact replay, HLO and downstream
score/reset consumers pass. The frozen original determinant cannot compile in
XLA CPU/GPU; this is retained as a graph comparator limitation.

Runs 03987, 03991--03997 and 04004 pass the focused checks and policy suite;
failed localization attempts 03988--03990 are preserved. The final policy has
253 guarded sources, 1,359 exact allowances, no violations and no stale
exceptions. The direct KDM/LEDH call-chain is guarded; an 80-module static
import overapproximation still has 17 optional/reference or metadata modules
outside the guard. This is audit debt, not a claim that the whole repository is
policy-clean. Candidate-only GenUT, latent-SIR, transport and metadata routes
remain open in F01--F20.

Descriptive stage cost arms 03998--04003 pass. CPU cold/warm medians are
original graph 1.143 s/3.642 ms, repaired graph 0.411 s/6.634 ms and repaired
XLA 0.824 s/0.903 ms. GPU 2 medians are 2.081 s/10.070 ms, 2.376 s/53.679 ms
and 1.493 s/4.736 ms. These single-process fixture costs are not a statistical
ranking or target-capacity claim. The KDM repeated-construction native memory
gap remains open; explicit bounded worker lifetime is qualified.

Charges through 04004: 84425.456798 CPU / 76718.624339 GPU seconds,
leaving 32.548484 CPU / 30.689271 GPU process-hours under
56/52-hour caps. This safety/stage unit used 158.904513 CPU / 155.801041
GPU seconds, including failed localization attempts 03988--03990. The complete
run files/source snapshots are hash-verified in
`ledh-safety-stage-evidence-04004.tar.gz` and its receipt. No worker is active.

Open master gaps remain: original cross-mode precision disposition, repeated
XLA native retention, DZ5 graph replay/GPU graph finite-difference failures,
external callback autodiff/pfor debt, public LEDH integration/reset qualification,
initializer/staged supervisor, reporting/isotropic cases, target capacity and
F01--F20 terminal dispositions. Main remains unmerged; canonical LEDH rebuild is
excluded by user direction.

Next: commit and push this safety/stage phase, then continue the remaining
transitive filtering/gradient call-chain audit under a new bounded plan. Do not
silently add broad allow-list waivers or promote optional modules.
