# Campaign supervisor history-memory repair

The supervisor now streams old run manifests and retains only budget fields and
the exact retry count while launching a worker. It previously retained every
historical source-hash dictionary. On the same frozen 3,870-record history,
three matched fresh-process repeats show a 98.54% reduction in sampled peak RSS.

| Loader | Median timed seconds | RSS before MiB | Sampled peak / retained MiB |
|---|---:|---:|---:|
| Original at 4c37f9f40 |4.76077|32.85|2576.76|
| Streamed |3.39314|32.98|37.55|

Every run preserves exact CPU/GPU cumulative charges (including supplemental
and unfinished-run reservations), next-run number 3871, and source-sensitive
attempt count. Separate tests reject changed/deleted source identities, retain
the three-attempt veto, and fail on malformed metadata. The terminal gate now
scans once and keeps qualified group names, while status and latest-record
reading also avoid retaining the entire history. The ordinary measurement
matrix may retain its selected qualifying results; this is not a universal
claim that all supervisor storage is constant-size.

The baseline is the exact original loader/accounting source from Git. A frozen
path/checksum index excludes newer campaign records without rewriting historical
manifests. Matched workers 03876--03881 share one current source closure and
three repeats per arm. Independent analysis verifies run/JUnit status, device,
fixture identity, all decisions and source identity before computing medians.
The 0.713 wall-time ratio is descriptive; disk cache and shared-host effects are
not controlled tightly enough for a broad speed claim.

Initial 03873--03875 remain preserved. Their retained-RSS measurements are
useful, but the process lifetime maximum includes earlier memory. After adding
in-operation sampling, the streamed helper's lifetime peak is already about
510 MiB before timing and stays there, while sampled/current RSS during loading
stays near 38 MiB. Therefore lifetime ru_maxrss is not this operation's peak.
This separates a measurement artifact from the real old 2.7 GB retention.
No claim about TensorFlow tensor allocation, compiler retention, device memory,
executable eviction or numerical target speed follows from this harness repair.

Receipt: `artifacts/filter-gradient-repair-20260917/driver-history-verification-03882.json`.
It binds the raw archive, reproducible analysis and frozen history index. The
underlying old manifests remain in the campaign archive chain/shared raw root;
the new index identifies their exact bytes. The current source and all twelve
unit workers are archived. Total unit charge is 106.057153
CPU seconds, below its 1,800-second limit. Final policy 03882 passes 129 checks;
Ruff and whitespace checks pass. No algorithm, comparison gate, or numerical
policy allowance changed.

| Decision | Evidence | Remaining uncertainty | Next action |
|---|---|---|---|
| Accept history-retention repair | Exact accounting/retry behavior and three fresh-process repeats pass | Selected measurement-result retention is separate | Use the streamed supervisor for subsequent work |
| Keep target-memory findings open | This workload does no target execution | CPU XLA and endpoint compiler-memory causes remain unresolved | Continue the endpoint capacity plan |
| Keep main unmerged | Graph-reference, public consumer, initializer and F01--F20 gates remain open | Whole-program completion | Continue bounded graph diagnosis and consumer repair |

Review: the first lifetime-peak reading could have falsely suggested a 500 MiB
loader allocation. Recording its pre-timing value and sampling current RSS
shows why that interpretation is unsupported. Exact frozen history/source checks
and adverse accounting tests constrain the more serious risk of silently
changing budget or retry semantics. New target kernels were not evaluated here.
