# Current XLA execution dispositions

Question: which active filtering/analytical-gradient call paths still violate
the execution policy or lack applicable validation after the scope correction?
Baseline is c4950a827. The corrected terminal queue controls scope; adaptive
iAPF and KDM are deferred, and unrelated historical optimizer research is not
a prerequisite. Fixed fitted-APF and shared helpers for other filters remain
in scope.

First refresh the syntax/call inventory, including implicit package imports,
then reconcile F01--F20 against current source, actual public consumers and
saved execution/numerical evidence. Use the recorded dependency bytes to reuse
unchanged results. Separate execution qualification from unresolved resource
acceptance and from scientific/canonical admission. Do not close a finding
from a syntax count, decorator, diagnostic result or a different score authority.

Run from /tmp/bayesfilter-filter-gradient-xla-validation-20260918:

```sh
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py audit --device CPU
```

This standard-library inventory imports no numerical implementation. The
existing runner records the exact command, source hashes, elapsed time and a
unique run directory under the campaign artifact root. Allocate at most two
300-second CPU workers (600 seconds total), including one infrastructure retry;
no GPU or numerical worker is needed. Reopen the gzip, verify its schema and
counts, and retain parse failures with their actual ownership. Preserve any
failed attempt. Stop if the scan cannot produce a complete inventory within
this allocation; do not turn it into numerical experiments.

For each suspected violation, inspect the operation and its enclosing caller.
Configuration/report traversal is different from numerical iteration. Record
an active violation only with a concrete reachable in-scope path. Unknown
callback dispatch remains an evidence gap. A newly confirmed runtime defect
gets a focused repair and test allocation before numerical execution. A
pre-existing diagnostic numerical failure stays recorded; it blocks this
rewrite only when tied to an affected usable result, score, decision or failure
contract, or an unsupported use that still needs an explicit block.

Deliverable: refreshed inventory and a per-finding disposition with source,
evidence, remaining execution/resource obligations and deferred/out-of-scope
parts stated separately. Update the master, checkpoint and ledger consistently.
No runtime, algorithm, numerical threshold, policy allowance, canonical claim
or merge decision changes merely because a finding is classified.

Primary-agent skeptical review: the previous inventory predates the latest
input-preparation and geometry boundary repairs; renewing it is relevant to
execution coverage. Its raw loop/NumPy counts include tests, reporting and
historical artifacts and are not defect totals. Static resolution cannot prove
dynamic execution, so current dependency and saved live-caller evidence must
support dispositions. This bounded source review avoids both blanket reruns
and the scope creep corrected by the owner. Existing memory/performance gates
remain unchanged. No subagents or additional numerical research are launched.

The refreshed inventory is run04940/audit.json.gz:7670 Python files,7669
parsed,2006 owned source/harness modules,31910 loop syntax sites and283 NumPy
import sites. The sole parse failure is the preserved historical vendor
leading-zero literal. Import overapproximation reaches418 modules from315
guard roots and identifies11 NumPy candidates for contextual review. These
counts are search leads, not defect counts. The unguarded-core candidate list
contains1368 syntax sites in104 files, including diagnostics, reporting,
configuration traversal and deferred code. It cannot establish104 violations.

The first confirmed missed active defect is the SSL-LSTM fixed-replay scorer's
Python time loop and eager public default, reached directly by the Phase6
benchmark that stamped XLA. Its focused repair and actual-caller qualification
are in filter_gradient_ssl_lstm_replay_execution_20260929.md. CPU/GPU runs04981
and04982 pass17 tests each after preserving original manifest coercions and
both original RNG contexts. Costs and memory follow-up are separate gates.

This per-finding work map supersedes treating19 broad open labels as19 known
unfixed numerical bugs. It is an intermediate disposition, not a terminal
compliance certificate. Entries identifying installed repairs still need
current dependency/evidence matching and applicable resource acceptance.

| Finding | Current implementation/evidence boundary | Remaining closure work |
|---|---|---|
| F01 GenUT loops and score provenance | Obsolete batch GenUT route retired; reduced-primal and analytical correction are distinct authorities, witnessed04936--04939. | Determine affected uses of saved reduced precision/cap failures; preserve explicit unsupported/canonical-admission blocks. No excluded canonical rebuild. |
| F02 Contract E parameter recurrences | Native/batched derivatives and stable compiled factory in ledh_contract_e_canonical_lgssm_tf; latent-SIR result20260926. Older Python date helpers have separate reference roles. | Match current dependencies to strict tests and actual factory consumers; do not infer canonical admission from the name. |
| F03 TT analytical adjoint | run_adjoint_score_filter delegates to make_adjoint_filter in squared_tt_native_adjoint_engine_tf; outer code checks completed status. | Reconcile saved adjoint and callback tests with current sources and complete-call costs. |
| F04 TT enclosing execution | Branch-axis and adapted-map factories enclose native fitting/time recurrences; Gaussian variant and frozen preparation have separate authorities. | Confirm actual benchmark consumers and shared-preparation dependency hashes; retain historical v0 as reference only. |
| F05 Actual-SV/fixed-adjacent TT | Native time/ALS/parameter control installed; scalar public value/score use make_scalar_adjacent_state_fixed_tt with XLA default. | Reconcile actual-SV analytical versus AD evidence separately; current caller/preparation coverage remains necessary. |
| F06 Frozen-proposal APF | FrozenProposalAPFProgram defaults to compiled evaluation; _evaluate_core uses tf.while_loop. | Match preparation, score and frozen-proposal caller evidence; this is distinct from deferred adaptive iAPF. |
| F07 General particle/LEDH loops | Shared numerical owners and public value/input/score repairs qualified through04913; bootstrap/OT/structural have existing focused tests. | Current caller coverage, fixed fitted-APF residency, streaming costs/capacity and GenUT affected-use disposition. Adaptive iAPF and KDM deferred. |
| F08 Dense/streaming LEDH score | Native execution and separate analytical authority retained; finite-program AD remains diagnostic. | Same applicable endpoint/resource work as F07; no canonical-score substitution; KDM-specific work deferred. |
| F09 SQMC/initialization | Native Hilbert exchange/sort control and initialization recursion repairs; ordering tests and transitive GenUT evidence exist. | Bind current initialization/RQMC consumers and exact stream/ordering evidence. |
| F10 Quadrature/grid preparation | TensorFlow quadrature replaces named NumPy rules; DNS and retained-moment consumers are guarded. | Match node/weight/order evidence and enclosing preparation owners to current sources. |
| F11 Retained moments/SGQF preparation | retained_moment_program has a fixed signature and native contractions; SGQF numerical preparation has focused tests. | Confirm heterogeneous-basis/configuration traversal dispositions and current complete caller evidence. |
| F12 Admission/identity NumPy | TF validation and standard-library/TF serialization repairs have identity/readiness tests. | Verify reachable admission/identity callers and exact encoding; scientific admission criteria remain unchanged. |
| F13 SGQF model derivatives | Predator-prey adapter calls explicit transition state/parameter Jacobians, replacing implicit pfor AD. | Match analytical RK4 and actual SGQF derivative-branch evidence; no sampler run needed. |
| F14 Other implicit pfor | Identified sites repaired/retired and qualified through04727. | Preserve scoped closure; final source integration must retain the guard. |
| F15 CPU score pool | Default requires batch-native shards; JIT defaults true; scalar route explicitly reference-only; TF serialization installed. | Check current construction sites and actual batched-shard tests; no NeuTra training is authorized here. |
| F16 Prior/warmup defaults | BGS prior wrappers have explicit signatures and XLA; warmup switch migration recorded. | Match current API/default and prior evidence; tuner/sampler mathematical requalification is outside this rewrite. |
| F17 Joint target/simulation | Native affine/time/substep recurrences installed; SIR simulator factory has fixed signature/XLA. | Confirm intended enclosing callers versus explicit validation/reference harnesses; do not expand into HMC validation campaigns. |
| F18 Reachable inference NumPy | Shared geometry/input/identity migrations and actual-DZ5/import-isolation evidence through04935 and04618--04631. | Finish current import/caller dispositions and applicable compiler/lifetime costs. Historical unselected optimizer trajectories are separate research. |
| F19 Stable enclosing signatures | Shared bounded owners cover numerous filter/score routes; SSL-LSTM omission is repaired and under final costs. | Complete current public-boundary audit; distinguish configured factories from eager endpoints. Mixed findings retain iAPF/KDM deferrals. |
| F20 Indirect NumPy control | Static TensorFlow control reduction migrated; source guard inspects more than import statements. | Verify actual streaming consumer and current helper evidence; import counts alone are insufficient. |

Next source-review deliverable is a source-bound evidence map for these rows,
reusing unchanged qualified runs and renewing only affected or absent checks.
Neither a stale historical passing test nor an unchanged top-level file alone
proves its current callback/dependency closure. Unknown dynamic dispatch stays
explicit. Resource triggers remain in the corrected terminal queue.

Import review of the11 candidates: generalized_sv_sgqf_tf and
native_generalized_sv retain local dense-reference quadrature imports;
sir_latent_preclip_reference_tf is explicitly an independent CPU reference;
hmc_tuning's local imports serve its diagnostic helpers. The two testing
modules explicitly contain independent model/Kalman oracles. The identifiable
SSL-LSTM geometry, minimal SSL-LSTM conditional-slice oracle and Phase5 tuning
harnesses each explicitly declare diagnostic/non-promoting roles. The
score-aware teacher projection declares an independent CPU reference.
run_contract_e_tp_scalar_sv_prefix uses NumPy Legendre quadrature only for its
dense-reference loop (lines247--260), with other NumPy arithmetic in post-run
comparisons (lines275 onward). No package/runtime import of these final two
benchmark modules was found by the current source search. This classifies
these imports in their inspected roles; it does not admit their numerical
outputs, certify sampler/training paths or establish arbitrary dynamic-import
purity. The conservative closure includes guarded diagnostic runners and
unexecuted local imports, so module reachability alone is not runtime use.
