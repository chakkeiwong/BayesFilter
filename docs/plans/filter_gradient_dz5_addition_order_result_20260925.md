# DZ5 addition-order diagnostic result

Completed diagnostic: the candidate is rejected under the
[bounded plan](filter_gradient_dz5_addition_order_unit_20260925.md).
No numerical runtime change has been installed. The diagnostic uses the unchanged
source snapshot plus a verbatim, checksummed executable overlay; module-file
checksums alone must not be presented as its executed source identity.

03904 and 03905 passed numerical cancellation/gradient checks, then failed XLA
compiler-IR export in the new diagnostic harness. The first failure came from
Python default arguments in the function signature; the second from TensorFlow's
implicit device inference attempting an unseeded random operation under op
determinism. A closure factory and explicit compiler-IR device `/CPU:0` repair
those harness defects without changing numerical input or computation.
03906 passes both graph and CPU XLA primitive tests, with one trace per
exposure case and exact repeated results/derivatives. Preserved failed workers
are charged; they do not qualify the original failed harness.

03907 passes the first actual-target overlay worker: original 185-row bank, 48
observations, normal arithmetic optimizer, five calls with reused inputs and
unchanged tuple outputs. Its first call passes both finite-difference checks
(maximum normalized errors 0.1085757785 and 0.1637037733); all four replays are
exact for values, scores, validity and branch status. Import/snapshot checks,
one trace and no-host-callback checks pass. The executed optimized graph retains
one `_noinline` function with exactly `AddV2, Identity`. The prior five-input
AddN is absent; two callback three-input AddN nodes remain. This nominates the
intervention for a full-target check; it does not prove all-call determinism.

03907 elapsed 608.197867 seconds. Cold/replay observations are 135.104736 /
110.938545 seconds; sampled peak RSS is 1,899,302,912 bytes. These include
diagnostic graph collection and are not matched performance evidence.

03908 passes both finite-difference gates for all 96 observations, but its
replay is not exact: values, validity and branch status are equal while 3,846
of 4,255 score entries differ, with maximum absolute difference
`9.43600753089413e-12`. The candidate therefore fails the unchanged exact-replay
veto and is rejected for installation. Independent analyzer output confirms the
48-step nomination and 96-step rejection from the saved raw artifacts.

Source inspection also maps the external credit callback's repeated parameter
and process-Jacobian additions to a plausible source of the optimized AddN
inputs. Its `noise_result` and `direct_result` accumulate two country contributions
at frozen `two_currency_double_zlb_credit_target.py:281--282`; the BayesFilter
transition tangent then adds propagated state/process terms. This is source
structure evidence, not proof of the failing operator. Exact graph/source
mapping is still needed. No external source file has been changed.
