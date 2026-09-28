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
