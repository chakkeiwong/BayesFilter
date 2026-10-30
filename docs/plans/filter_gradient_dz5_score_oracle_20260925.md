# Fresh DZ5 analytical-score oracle

Continue E5 under the existing 56 CPU / 52 GPU hour campaign caps. The LEDH
checkpoint is committed and pushed as `4c37f9f40`; charges through 03862 leave
34.9372 CPU / 30.9988 GPU hours. No MacroFinance source changes, admission
refresh, training, sampler restart, package changes or main merge are included.

Question: does the current analytical score compute the derivative of the
unchanged CDF value target for every one of its 23 parameters? Prior graph/XLA
and archived-source parity cannot rule out a shared derivative defect. Repeat
the independent five-point oracle from the frozen MacroFinance
`scripts/qualify_dz5_cdf_runtime.py:174`, with its comparator at line 111.
Use the identical truth-centered 185-row bank, prior-scale perturbations at
steps 1e-3 and 5e-4, the original 96 observations, and unchanged tolerances
`atol=1e-8, rtol=1e-7`. The fixture identity remains
`e116fe853c8579369036ab2ce57724ba07544524d0920fa0a706d76714f75d8a`.

Create a new versioned read-only snapshot from committed BayesFilter package
bytes and the unchanged frozen MacroFinance closure/inputs. Record the original
qualifier checksum and audit every actual project import before and after target
execution. Retain the native import prerequisite as an import-only dependency;
forbidden callbacks in the graph still veto. Old admissions remain stale.

Run the graph reference and default-XLA target in separate fresh processes,
first CPU, then GPU if CPU qualifies. Each uses an explicit stable 185-row
signature, TF32 disabled and verified memory growth on GPU. Test the full bank,
both finite-difference refinements, finite values/scores, all valid statuses,
exact replay, one trace, and enclosing HLO/no callbacks for XLA. Save inputs,
values, every analytical/finite-difference coordinate and normalized residuals
before enforcing the numerical gate. NumPy here is an independent diagnostic
oracle only; no finite-difference or autodiff score enters runtime code.

Reserve at most eight workers / 7200 charged seconds for this unit, including
localized retries and final policy renewal, within the remaining global caps.
Use the stable runner, 900-second outer / 840-second isolated child bounds, and
one numerical worker at a time. Do not edit runtime/scripts/tests while a worker
is active. Source drift, missing device provenance, unavailable growth or an
unexpected numerical failure stops that arm and triggers a bounded diagnosis.
At most three localized retries per unchanged fixture. A failed numerical
oracle cannot be reclassified as passing or repaired by tolerance relaxation.

Record cold/replay time, sampled current and peak host RSS, and allocator
availability. These are capacity observations in separate graph/XLA processes,
not a repeated performance comparison or proof of compiler-memory causation.
Preserve results in numbered campaign directories and a checksum receipt.
If both backends pass, compare their saved banks, status and score records at
the original tolerances. Do not issue a new adapter admission from this unit;
public initializer/staged supervision and other E5 gates remain separate.

Skeptical review: the oracle differentiates the same scalar target but obtains
the derivative independently from values; it is stronger than score self-parity.
Two step sizes constrain truncation/roundoff, but only this frozen local scope
is tested. Original target truth is deliberately reused to renew that exact
engineering check; no claim of posterior-wide gradient validity follows.
Separate processes avoid the earlier combined graph/XLA memory ambiguity.
Baseline, raw errors, import provenance, stop conditions and budgets are explicit.
The plan passes this bounded self-review; no independent agent review was used.

03863 graphCPU passes the original five-point gates (maximum scaled errors
0.55736/0.36804) and valid statuses, then fails exact score replay:3911/4255
entries differ, at most1.09139364e-11 absolute /1.42845e-13 relative. Preserve
this failed arm; exact replay is not relaxed. The next localized retry enables
TensorFlow op determinism explicitly before target construction and saves every
replay value/score/status before enforcing equality. This is an execution-setting
diagnosis; the target, input bank, oracle, hardware class and budgets are unchanged.
Apply the same explicit setting to the remaining cohort and record it. If replay
still fails, preserve it and localize the operation before another unchanged retry.
No claim that this setting caused or solves the discrepancy precedes evidence.

Terminal analysis must recompute both finite-difference arrays directly from the
saved185-row value bank using Python standard-library scalars, independently of
the NumPy test report. Validate run/JUnit pass, source/snapshot/qualifier identity,
actual loaded-module closure, exact bank and status identity, full graph/XLA and
CPU/GPU values/scores at the original1e-8/1e-7 gate, exact saved replay and XLA
HLO hashes. Reject a changed score/value row, a false pass flag, stale snapshot,
missing replay and invalid row in diagnostic mutation checks. A saved success
flag is not an independent oracle. Timings/RSS remain descriptive single-process
capacity observations even after numerical and provenance gates pass.

03864 preserves a second CPU graph replay failure even with op determinism:
values are bitwise identical; scores differ by at most1.318767317570746e-11.
The five-point gates still pass. The determinism flag does not repair this case,
so do not retry unchanged or claim causation. Next run the independent CPU XLA
arm on the same185-row bank/settings to separate default-XLA behavior from the
explicit graph reference. The graph arm remains vetoed; success of XLA cannot
retroactively pass it. Localize graph replay on a smaller horizon with saved
per-stage outputs and thread/optimizer controls before another full graph run.
This third worker remains within the eight-worker allocation. Source/runtime
code is unchanged; the resulting evidence is endpoint-specific, not full E5
qualification or permission to loosen replay.

03865CPU XLA passes both original five-point gates (max scaled0.39172/0.76784),
finite/status checks and bitwise value/score replay, then its diagnostic HLO
export fails. TensorFlow's `compiler_ir.maybe_get_device_name` generates a random
scalar to infer placement; op determinism requires a seed, even though the
numerical target contains no random operation. Supply the already fixed device
name to compiler-IR export, avoiding this diagnostic RNG. Preserve03865 as a
harness failure; repeat the complete worker with the same target/bank before
qualifying XLA. This is localized infrastructure retry4 of the8worker unit,
not a change to the target random stream. The CPU graph replay veto remains.

03866CPU XLA passes the complete oracle, replay, one-trace, HLO and loaded-source
checks after the export fix. Proceed to GPU XLA under the same controls; the
CPU graph arm remains separately failed. A passing XLA numerical path permits
qualification of that path on GPU; it does not upgrade the graph reference.

03867GPU XLA passes the complete oracle/replay/HLO/source checks. Independent
standard-library reconstruction passes the CPU/GPU records and rejects six
corrupt-record mutations.03868GPU graph is preserved as a numerical failure:
the second finite-difference step has max scaled error1.00448179 (unchanged gate1).
Do not relax it or present its timing as a qualified speed comparison.

Use worker7 for a bounded CPU replay localization, two sequential isolated
children at intra-op thread counts2 and1. Keep the exact185-row bank and model
prepared from the96-observation fixture, but run only its first2 observations
through the shared rectangular filter. Compare model initial values/tangents,
final values/scores, means/factors and their tangents over three repeated calls.
Save full diagnostic arrays and summarize maximum errors/changed counts. These
short-horizon observations explain replay only; they cannot qualify the96-step
oracle or establish a derivative cause by themselves. No copied recurrence or
numerical method change. This diagnostic is explanatory regardless of whether
replay differs. Final policy is worker8; any further repair requires a bounded
follow-up within remaining global authorization, not unrecorded retries.

03869's two-observation diagnostic passes exact replay on both1 and2CPU threads
for initial inputs/tangents, values/scores and final moments/tangents across
three calls. It does not reproduce the96-step failure, so thread count has not
been identified as its cause. A follow-up must locate the first differing
longer prefix or freeze later-step operands before claiming an operator cause.
The original two full graph failures remain vetoed.03870 renews policy as the
last worker of this eight-worker unit; no further numerical retry is included.

An80-digit Decimal recombination of the saved GPU graph values does not rescue
the failed smaller-step comparison: normalized error grows from1.00448to1.09914.
Saved-scalar half-ULP sensitivity is1.0914e-8 versus allowed2.5622e-8 for that
coordinate; this is explanatory sensitivity, not a bound on the unknown target
value error. No rounding-based waiver is installed. The diagnostic JSON is
`artifacts/filter-gradient-repair-20260917/dz5-score-rounding-diagnostic-03868.json`.
