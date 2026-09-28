# Terminal source review progress

The r2 AST scan completed in129.212 seconds and its compressed JSON was
reopened and parsed successfully. The r1 scan timed out at120 seconds during
gzip output; its partial artifact is unqualified. Both are charged once,
249.212 CPU process-seconds total. Using gzip level1 changes only compression;
audit rules and JSON content are unchanged. No numerical test ran in this scan.

The current inventory discovers7579 Python files, of which7578 parse. One is
the new untracked renewal harness at scan time;7578 are tracked. The sole parse
failure remains the vendored historical
`experiments/student_dpf_baselines/vendor/2026MLCOE/old_pt1_submission/filters.py`
leading-zero literal. The inventory contains1997 owned source/harness modules,
31807 loop syntax sites and284 NumPy import sites across that broad scope, plus
240970 best-effort static call edges. These counts include control, preparation,
reporting and diagnostics; they are not violation counts. Archived artifacts
account for3771 Python files and are kept as historical evidence.

Following imports from the276 guarded sources reaches330 modules. Four
NumPy-containing modules appear in this overapproximation:

| Module | Inspected use | Current disposition |
|---|---|---|
| `highdim/generalized_sv_sgqf_tf.py:261` | Local NumPy import in `generalized_sv_dense_value_reference_status` for independent Hermite quadrature | Explicit reference boundary; do not treat the mixed module's import count as a runtime violation |
| `highdim/native_generalized_sv.py:355` | Local NumPy Legendre rule in dense reference preparation | Independent reference; current exact guard retains its scope |
| `testing/nonlinear_models_tf.py:694` | `_dense_gaussian_quadrature_points` calls local `_hermgauss` reference helper | The guarded adapter imports `model_b_observations_tf`, not the dense projection helper; dynamic/external calls remain outside this static claim |
| `inference/hmc_tuning.py:1772,1959` | Gaussian dual-averaging diagnostic and historical fixed-trajectory diagnostic inspect discarded draws with local NumPy imports | Diagnostic helpers; this does not qualify the rest of the tuner or prove an arbitrary external callback pure |

Evidence is `terminal-source-audit-20260928-r2/guarded-transitive-numpy.json`
under the raw campaign root. Static imports include unexecuted branches; the
inspected function context is essential. Dynamic imports and callback dispatch
remain limitations. No new allowance or runtime reclassification is installed.

Outside the guard, the registered single-cloud LEDH value entry still resolves
to `ledh_canonical_filter_tf.canonical_value_and_diagnostics`, with the already
recorded NumPy seeded inputs and host numerical recurrence. The bounded native
value/score work is described in `filter_gradient_ledh_native_endpoint_20260925.md`;
its owner-interface component qualification did not migrate the registered
consumer. That existing endpoint gap remains open. This review neither rebuilds
canonical LEDH nor upgrades a reduced score to canonical status.

`dense_directional_score_geometry.py` explicitly identifies its finite-radius
directional diagnostic role. Current package/scripts text search finds only
its definition module, not an admitted runtime consumer. This is a limited
call-site observation, not proof about external users. Other unguarded NumPy
modules, public compilation boundaries and callable-factory paths still need
their contextual terminal dispositions.

| Decision | Primary criterion | Veto/uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Retain the current inventory | Gzip/schema readback passes; all discovered files represented including parse failure | Static resolution cannot prove execution | Reconcile endpoint evidence and remaining unguarded paths | Whole-repository compliance |
| Preserve F01--F20 as open | No complete terminal consumer/evidence matrix yet | Known registered LEDH endpoint and strict numerical/lifetime gaps remain | Finish source/current-consumer review without re-running unaffected suites | Whole-program completion or permission to merge |

Post-run review: the large inventory mainly expands archived/test coverage.
Treating its raw syntax total as migration debt would be wrong. Conversely,
passing the276-source guard cannot cover the remaining public endpoints. The
next useful step is endpoint-specific evidence reconciliation, not another
unchanged whole-repository scan. No independent reviewer was used.

The saved-cost readback completed in4.775 seconds and correctly returned a
nonpassing report:0 current generic measurement pairs,1116 missing requested
pairs,770 excluded older records. Of those,763 have stale harness hashes and7
use an outdated schema. Every stale record differs in
`measure_filter_xla_memory.py` and `filter_repair_benchmark_worker.py`; some also
have changed fixture sources. This is not a measured numerical failure and
does not prove1116 new jobs are necessary.

The generic comparator only reads `action=measure` records, ending at01481 in
this campaign. Later matched costs live in endpoint-specific pytest artifacts
and their qualified readbacks. The terminal master therefore needs an evidence
index that reconciles those result families and their actual source closure;
the old generic comparator cannot represent whole-program completion. Preserve
its stale exclusions. Before renewing any numerical cost, inspect whether the
later endpoint result already answers it with current dependencies and valid
hardware/input/timing provenance. Do not translate stale historical harness
bytes into a blanket equivalence exception. The raw report and differing-file
detail are in `terminal-cost-readback-20260928-r1`.

The first static closure omitted Python's implicit parent-package execution.
Following those edges expands the current guarded-root closure to354 modules
and11 NumPy-containing candidates. Seven additional reference modules are
reached through the eager `bayesfilter.adapters` initializer, including the
NumPy Kalman, particle, sigma-point and derivative references. Importing the
guarded TensorFlow BGS submodule executes that initializer first. This is a
confirmed source-level reference-isolation defect, not permission to use those
reference implementations as runtime code.

A minimal lazy-export repair is prepared in `bayesfilter/adapters/__init__.py`.
It preserves every public export, its original order, and the three renamed BGS
bindings. Static old/new export comparison, an implicit-parent/lazy-alias audit
fixture, and the exact source guard pass. The guard adds this package initializer
without adding any policy exception. The audit now emits implicit package edges
and resolves lazy alias attributes. A patched-source closure has345 modules and
the same four explicit diagnostic/reference candidates discussed above. Real
fresh-process BGS import interception and frozen prior value/score checks remain
queued after renewal; the repair is not qualified by static checks alone.
The exact before/after import chains are saved in
`terminal-adapter-import-review-20260928-r1/parent-closure.json`.

The new standard-library endpoint index now reads both row-local digest pairs
and named artifact tables. It also reopens the committed single-locator GPU
renewal archive, verifies its full archive and report hashes, and indexes the
saved analysis without launching a new worker. This recovers the qualified
04363--04368 cohort omitted by the old generic reader. The first index is
preserved as partial format discovery; r2 records its reader source and hash.
Both attempts are charged (0.966086 and1.066799 CPU seconds), well within the
separate600-second readback/integrity-test allocation.

The r2 index covers22 reports across ten endpoint families. No recorded file
hash mismatch was found. GenUT's report does not record row-local artifact
hashes, and the remaining-SVD receipt has no direct run references; those are
explicit index limitations requiring their existing analyzers/receipts, not
newly passing evidence. Most old run source dictionaries cover thousands of
files rather than actual dependency closures. The index therefore records
each current source delta without treating an unrelated change as a measured
regression or allowing it to waive an affected dependency.

For the current posterior-initializer cost report, eight recorded source
paths have changed: two numerical geometry dependencies, the adapter package
initializer and five harness/policy/test files. The two numerical changes
need contextual cost renewal review even though fresh actual-target and
lifetime checks pass. For the single-locator cost reports, the locator
implementation itself remains unchanged; the broad source differences alone
do not justify repeating its complete matrix. This is an evidence-index
result, not a terminal cost acceptance or whole-program completion claim.

Fresh-process adapter qualification04629 passes12 checks in23.962 seconds.
The package, direct BGS, public BGS and alias cases install an import blocker
for the repository NumPy-reference modules; all succeed without loading those
modules. Explicit MacroFinance reference access remains functional. The exact
43-export map/order, unknown-name behavior, three BGS aliases and existing
frozen prior/analytical-score assertions pass. This is actual import and prior
execution evidence. Full policy04630 passes160 checks in9.584 seconds. The
adapter unit closes after2 of4 workers and33.547 CPU seconds, with277 guarded
sources and no added policy exception. Other reference/runtime boundaries and
all broader F18 obligations remain separately open.

Evidence-index integrity tests04631 pass all6 cases in3.874 seconds: changed
runtime sources remain visible without automatic acceptance, corrupt result
and manifest digests fail, absent evidence stays explicit, named artifact
tables are verified, and archived reports require archive/member identity.
The r2 readback verifies348 reference entries representing282 distinct runs;
14 GenUT run/reference identities still lack row-local digest evidence in
that report. This closes only the bounded index implementation unit (5.907
CPU seconds including both readbacks), not the contextual cost review.

Terminal review of this repair: the independent frozen export map and real
import blocker address accidental API drift and reference imports. The
strongest remaining limitation is dynamic/external dispatch outside the tested
BGS routes. A newly reached reference module or changed prior result would
invalidate this scoped conclusion. The index's successful digest checks prove
preservation of evidence, not mathematical validity or current-source coverage.
