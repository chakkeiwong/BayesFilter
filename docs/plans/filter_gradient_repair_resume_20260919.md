# Filter and gradient repair recovery

Current checkpoint through 03975, September 26: native KDM execution repair
is committed and pushed at `6000ae63a`. The auxiliary has a stable-signature
XLA owner and native time recurrence, and the shared reset uses a native
Sinkhorn recurrence. CPU/GPU complete records, derivatives, rejection behavior,
changed operands, exact replay, public API and empty horizon pass. The final
30-check component suite and 18 matched cost workers pass. The latest policy
worker 03975 passes 129 checks (246 guarded sources / 1354 exact allowances).

The precision follow-up is complete: the original and repaired FP32 XLA reset
outputs are identical and match an independently checked FP64 reference from
the same rounded inputs. Maximum unchanged-tolerance units are CPU XLA 0.321
versus eager 1.025; GPU XLA 0.062 versus eager 180.983 with TF32 and 1.550 with
TF32 explicitly disabled. The original eager comparator is inaccurate at this
tolerance. Condition numbers around 60--153 do not establish severe
ill-conditioning. Original cross-mode failures remain preserved as an open
terminal disposition; no tolerance, ridge, dtype or default changed.

Repeated XLA construction retains native host memory despite collection of
later Python graphs and constant registered-function counts. Six builds grow
RSS to 2556 MiB CPU / 2472 MiB GPU; graph controls level off at 834/1363 MiB.
Live GPU allocation returns to 7168 bytes. Allocator trimming reduces RSS but
leaves the slope (CPU 1069 to 2428 MiB after trim). A single retained owner is
stable over 20 calls. Exact native ownership is unresolved. See the
[precision/memory result](filter_gradient_kdm_precision_memory_result_20260926.md)
and [next repair plan](filter_gradient_kdm_remaining_gaps_20260926.md).

Charges through 03975: 83915.367867 CPU / 76355.603728 GPU seconds,
leaving 32.690176 CPU / 30.790110 GPU process-hours under the
unchanged 56/52-hour caps. The extra 24 CPU hours are already included. This
follow-up used 267.818416 CPU / 179.780318 GPU seconds, including two policy
checks (12 planned CPU workers plus the final 8.633437-second policy check;
6 GPU workers). No worker is active. The full 03957--03975 raw files and source
supplements are verified in `kdm-followup-verification-03975.json` and its archive.
Earlier 03904--03956 evidence remains in the linked two prior receipts.

DZ5 default graph replay/GPU graph finite-difference failures remain open.
Its addition overlay is rejected; explicit arithmetic-optimizer-disabled CPU
reference 03901 remains qualified. Frozen external callback autodiff/pfor debt
remains. Public LEDH value/score integration, reset qualification, public
initializer/staged supervisor, reporting/isotropic cases, target capacity and
F01--F20 terminal dispositions remain open. Two reporting proposals await
previous user answers. Main is unmerged; no canonical LEDH rebuild is authorized.

Worktree `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`, branch
`repair/filter-gradient-xla-validation-20260918`. Raw root:
`/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917`.
One numerical worker at a time, with runtime/scripts/tests frozen during it.
No subagents, MacroFinance edits, package changes, CDF restart, stale admission
refresh, historical NeuTra promotion or premature main merge. Canonical NeuTra
remains the shared author-profile IAF. Remote main `5e16df06f` is integrated;
the repair branch was pushed after a clean fetch with no conflicts.

Next: execute the native-ownership unit in the remaining-gaps plan: compare
allocated/free/mapped native bytes and nested compilation ownership before
changing runtime ownership. Preserve mutable callbacks and the existing
explicit retained-owner path. Then continue the master consumer/terminal repairs.
