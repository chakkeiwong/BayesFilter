# Filter and gradient repair recovery

Current checkpoint through 03986, September 26: the completed native KDM
repair (`6000ae63a`) and precision follow-up (`04643213e`) are pushed. The native
ownership follow-up is complete. CPU/GPU allocation categories show roughly
252/200 MiB additional live host heap per rebuilt XLA owner. Compiling mixture
functions only through their owner does not resolve the slope; no such runtime
change is installed. Python graph collection and low live GPU allocation do
not free this native retention. Installed TensorFlow headers document a cache
without eviction; runtime logging confirms one new compilation per owner but
does not expose its exact live allocation owner.

An explicit bounded process lifecycle passes complete-record comparisons and
20 exact replays per retained owner on CPU and GPU, with changed callbacks and
confirmed child termination. Reuse one owner while callbacks/configuration are
fixed; changing tensor operands does not require reconstruction. General
repeated-constructor safety inside a long-lived process remains open. See
[the ownership result](filter_gradient_kdm_native_ownership_result_20260926.md).
03986 passes all 129 policy checks (246 sources / 1354 exact allowances).
The raw data, installed headers and hash-verified executed source revisions are
archived in `kdm-native-ownership-evidence-03986.tar.gz` and its receipt.

Charges through 03986: 84266.552285 CPU / 76562.823298 GPU seconds,
leaving 32.592624 CPU / 30.732549 GPU process-hours under
56/52-hour caps. The additional 24 CPU hours are already included. This unit
used 351.184418 CPU / 207.219570 GPU seconds including all child wall time. No worker
is active. One numerical worker at a time; freeze numerical sources during it.

Precision remains as recorded through 03975: original and repaired XLA are
identical and independently accurate at the unchanged tolerance; the original
eager comparator is inaccurate, largely because of GPU TF32. No severe
ill-conditioning is established. The original cross-mode terminal disposition
remains open. DZ5 graph replay/GPU graph FD, external callback autodiff/pfor,
public LEDH integration/reset qualification, initializer/staged supervisor,
reporting/isotropic cases, target capacity and F01--F20 dispositions remain open.
Two reporting proposals await previous answers. Main remains unmerged.

Worktree `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`, branch
`repair/filter-gradient-xla-validation-20260918`. Raw root:
`/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917`.
No subagents, MacroFinance edits, package changes, CDF restart, stale admission
refresh, historical NeuTra promotion or premature main merge. Canonical NeuTra
remains the shared author-profile IAF; canonical LEDH rebuild is excluded.

Next: execute [the safety/stage call-chain plan](filter_gradient_ledh_safety_callchain_20260926.md).
Remove the Cholesky rank loop, preserve squeezed mask shape, explicitly reject
nonfinite factors, repair stage reference recurrences and guard the full shared
helper closure. Then continue the master consumer and terminal repairs.
