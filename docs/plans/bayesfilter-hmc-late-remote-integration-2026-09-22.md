# Late remote integration and confirmation reset

While the large existing Git history was uploading, remote main advanced from
3582b4ac5 to c7adbda7d. The new commit repairs preserved-transition ESS and
percentiles, adds explicit metric-window preparation and progress records,
repairs batched local-center handling, and adds a TensorFlow particle reference.
The push correctly rejected a non-fast-forward update; no history was forced.

The merge conflicts with earlier module extraction and shared diagnostic math.
Retain the extracted architecture. Move incoming window-stage changes to
hmc_mass_adaptation, preserve the existing bootstrap/recovery path, and integrate
the optional explicit window without weakening metric gates. Keep the more
complete existing progress serializer and preserve incoming result events.
Keep shared corrected rank/R-hat arithmetic. Preserve the incoming named
Stan/ArviZ ESS recursion and its independent reference tests, with explicit
estimator/version metadata; the separately named TFP mean/quantile precision
baseline must not silently become that different estimator. The book must
describe those conventions accurately. Preserve the independent particle and
local-center repairs and test their affected paths.

Skeptical audit found that launching the frozen M21 HMC confirmation as
"current-source" evidence after this merge would be stale. Its waiting
coordinator was stopped before any confirmation worker started. The 272-fit
and 1024-null inventories remain historical prelaunch records. Rebind fresh
confirmation to a new merged snapshot only after regression checks and a small
Gaussian/beta-binomial public parity/cost pilot. Preserve completed M21 AR(1)
and M22 pilots with their exact old source identity. They remain valid baseline
evidence and are not relabeled as merged-source results.

Engineering acceptance requires all conflict markers removed, no reintroduced
monolithic duplicate implementation, public preparation/checkpoint/batching
regressions, ESS/rank/tail checks against the installed independent reference,
and the incoming particle tests. Record any intentionally changed diagnostics
and unchanged numerical invariants. Reserve up to 1800 CPU seconds from M19's
remaining 7200-second allowance for merge checks and up to 1200 seconds in
M21's revised 79200-second allocation for the fresh pilot. These are engineering
ceilings, not numerical defaults; campaign totals remain unchanged. GPU work
remains pending free permitted capacity. Only two numerical workers may run.

After checks pass, commit the resolved merge in the isolated checkout, preserve
the shared worktree's unrelated edits, fast-forward main, commit the tested
continuation changes narrowly, push normally, verify remote main, then remove
the temporary transfer branch. Keep source snapshots fixed throughout. Review
the final merged book and refresh the execution plan before restarting the
confirmation queue. Neither successful Git integration nor reference parity
closes posterior calibration, global-mode exploration or learned-map quality.

The tfgpu environment lacks ArviZ and matplotlib, so the incoming reference
tests would skip there. Use a disposable system-site-packages virtualenv under
/tmp/hmc-reference-20260922 with ArviZ 0.21.0 and its dependencies, recording the
resolved versions. This is an isolated diagnostic environment; it changes no
project or conda packages. Inspect its official ESS implementation before
interpreting parity. Tests must actually run; an optional-dependency skip is
not a passing reference check. CPU-only settings apply before TensorFlow import.

## Integration result

The resolved merge is `34cc0db09`; the bounded continuation commit is
`f9c86f41a`. The final affected-path inventory contains 354 passing named tests
and one skipped optional saved-DZ5 fixture. The ArviZ 0.21.0 tests actually ran,
including short/odd lengths, positive dependence, antithetic chains, ties,
per-chain MCSE, XLA parity and large-location invariance. The reference source
and its SHA-256 are preserved locally and in `m19-r1/arviz-source-audit.json`.

The first merge test exposed an actual integration error: a conditional local
`dataclasses.replace` import shadowed the module import on the explicit-window
path. Removing that local import repairs the path. Tests check both metric
evidence policies and preserve nondefault probe/recovery settings. A mock
stage fixture also needed its actual result interface's `payload()` method.
The rank-reflection test's fixed absolute tolerance failed by 8.53e-14 in two
tail entries. Its tolerance now propagates machine epsilon through the inverse
normal derivative; production rank arithmetic is unchanged. Independent rank
and R-hat reference checks remain in place.

Method identity is recorded in every sequential configuration, including when
additional assessment settings are absent. A checkpoint regression verifies
that changing this identity rejects reuse. This closes the otherwise missing
default-policy checkpoint binding. The 152 continuation checks then passed in
the isolated checkout without the shared workspace's unrelated q20 and training-
protocol changes. All failures and retries remain charged in the new ledger.

The new ordinary Gaussian and beta-binomial pilots completed on the committed
snapshot. They retained 23 and 17 verified siblings and assessed their
predeclared L=3 members, with independent fixed-count comparators. Worker times
188.58306595892645 and 227.3817994639976 seconds remain below the confirmation
reservation per fit. Their evidence and the estimator follow-up are in the
merged-source continuation result. No old result was relabeled.
