# Structural A/B documentation and Git closeout

Request: fully document the completed Model A/B surrogate-HMC experiment, commit the campaign work, merge remote changes, resolve conflicts and push. Target branch: structural-ledh-model-ab-20261023; merge source origin/main because the current branch has no upstream. Preserve unrelated route_ledger.json changes and other agents' worktrees.

Reader: a researcher familiar with Gaussian filtering and HMC. The account must explain the structural innovation proposal, exact versus finite-filter acceptance targets, NeuTra whitening, conditional reset precision, adaptive stopping, actual results and limits without requiring an appendix. Apply write-linear-scholarly-narrative; human reader feedback remains pending and does not prevent the requested update.

1. Audit model definitions, endpoint/force call chain, selected controls and saved terminal results.
2. Add the completed experiment to the structural-dynamics chapter; cross-reference from the surrogate-HMC chapter without relabeling its earlier five-model experiment.
3. Run focused campaign regressions with GPU deliberately hidden; build the LaTeX and inspect the added pages. Preserve full logs locally and a compact audit record.
4. Commit campaign code, tests, documentation and reproducibility evidence, including raw posterior draws, map checkpoints and logs. Omit only 21 large, exactly reconstructible particle-design JSON files; preserve their hashes and reconstruction script. Fetch, merge origin/main, resolve actual conflicts, recheck affected claims, and push the current branch with upstream tracking.

Evidence contract: documentation must match saved data, target definitions, parameter ordering, diagnostic values, precision policy and code. Comparator is the archived campaign plus current source; primary pass is no confirmed discrepancy. Build failure, inconsistent numerical claims, unsupported exact-posterior claims, or failed focused regressions trigger repair before publishing. Runtime and prose length are explanatory only. No new stochastic ranking or global numerical default is claimed. No additional GPU experiment is required.

Skeptical review: preserve Model A exact endpoint versus Model B finite LEDH target; avoid labeling convergence as approximation accuracy, truth coverage as calibration, or the legacy verification storage name as extra sampling. Record Model B chain-four acceptance 76.8%, conditional FP64 scope, the chosen architecture's own-target training history, and the unresolved nonlinear posterior reference. Check claim-bearing consumer calls rather than only helper existence. The plan passes with these limits.

Defaults: use existing main.tex, chapter style, and current-branch remote integration. These are repository conventions, not scientific choices. Scientific settings come from saved selections/configurations. Protected prose baselines are the two chapter files at base commit f97f53c3851bd8e5192f48a11d0d83eb27580726. The edit adds results and cross-references; it does not remove existing equations or historical findings.

## Completed audit

The added structural-dynamics section explains the two model laws and priors,
the innovation-mixture importance weights, shared total-score recurrence,
compact Cayley identity, force/endpoint distinction, NeuTra training, conditional
reset precision repair, reference checks, adaptive sampling, physical posterior
intervals and reproducibility. The five-model chapter cross-references this
extension without changing the earlier experiment. The two existing chapter
edits are additive: no prior equation, citation or finding was removed.

Call-chain inspection traced StructuralTarget through the shared batched-score
consumer to structural_proposal, shared map construction, weight evaluation,
reset_cloud and moment correction; the frozen-map HMC endpoint/force adapters
were checked separately. The regression suite includes executable dense/compact
action and tangent parity, structural batch/full-filter value-score checks,
finite-difference scores, reset-repair no-fire cases, mapped diagnostics,
energy reporting and retained-sample semantics. The exact final Model B
selection's 723 recorded source hashes all match the current code. The 36
terminal warmup/inference tensors match their manifest hashes.

The focused CPU suite passed 75 tests (two dependency deprecation warnings,
51.83 seconds). GPU devices were deliberately hidden. Full logs and a compact
machine-readable numerical audit are committed under the campaign's
documentation-closeout directory. No additional GPU experiment was run.

The full main.tex builds with pdflatex and BibTeX to 781 pages. Cross-references
are resolved; the final build requests no rerun. Rendered PDF pages 138–145
(printed pages 116–123) were inspected, including the three numerical tables,
equations, running headings and reproducibility paths. Long headings and paths
found in the first render were repaired. The added section has no overfull
boxes. Existing layout warnings in other chapters remain outside this edit.
The local rendered PDF is at
/tmp/ledh-structural-test-design-20261023/latex-build/main.pdf; its SHA-256 is
recorded in the committed audit. Human reader feedback remains pending.

Evidence preservation includes raw posterior tensors, training checkpoints,
failed attempts and logs. Only 21 generated particle-design JSON arrays,
185,528,660 bytes in total, are excluded from Git by a campaign-local ignore
rule. All 21 were independently reconstructed in a fresh temporary directory
with TensorFlow 2.19.1 on CPU and matched the original byte hashes. The archive
includes the inventory, restoration script and instructions. Original files
remain locally; no prior evidence was deleted.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Documentation ready for requested commit and push | Code, saved numerical results and target definitions agree | Focused tests and full document build pass; source and tensor hashes match | Model B exact-posterior accuracy; reader feedback on prose | Integrate origin/main and push the current branch | New scientific default, statistical superiority or exact nonlinear posterior validation |

| Inference status | Finding |
|---|---|
| Hard veto screen | Terminal numerical-health checks pass; historical failed attempts remain labeled |
| Statistically supported ranking | None claimed |
| Descriptive-only differences | Acceptance, ESS, timing and point-score differences |
| Default readiness | No global algorithm or precision default promoted |
| Next evidence needed | An independent, refined posterior comparison for Model B if that accuracy question is pursued |

Post-audit red team: the main misleading interpretation would be to call Model
B exact-posterior inference because its R-hat is small. The manuscript explicitly
defines its finite-filter endpoint and records the unresolved reference error.
The exact Model A mean check is identified as being in q coordinates, distinct
from the physical-coordinate interval table. Chain-four acceptance of 76.8%
is disclosed. ESS and finite energy errors remain explanatory diagnostics.

## Commit-hook environment repair

The first commit attempt failed before oracle execution: the repository hook
activates tftwogpu, whereas the campaign worktree's native Sylvester operation
was built against tf-gpu. Collection reported an undefined Abseil symbol when
loading that shared library. This is a TensorFlow/native-library ABI mismatch,
not an observed numerical oracle failure.

The existing tftwogpu build in /home/chakwong/BayesFilter has identical
symmetric_sylvester_op.cc and Python wrapper source. Using that compatible
operation through the existing BAYESFILTER_OP_LIB_DIR override restored test
collection (one canonical-stage test collected). A private copy under
/tmp/ledh-structural-test-design-20261023/hook-compatible-ops freezes the library
for the commit hook; its SHA-256 is
661f11b9db1f6e9ab9ce4aae8ae86591779cdbdfe6fda777a7c87956ec2686fa.
GPU devices remain deliberately hidden. No campaign library, shared hook,
numerical implementation, or oracle requirement is changed or bypassed. The
full oracle hook executes at the retrying commit boundary; its command log is
/tmp/ledh-structural-test-design-20261023/git-commit-compatible-hook.log.
