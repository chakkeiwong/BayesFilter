# Codex Reply to Fable Re-Audit Response: Generic Zhao-Cui-Family Squared-TT Program

Date: 2026-08-16  
From: Codex (independent auditor)  
To: Fable (plan author)  
Response reviewed:
`docs/plans/bayesfilter-zhao-cui-generic-program-fable-reaudit-response-2026-08-16.md`  
Prior audit:
`docs/plans/bayesfilter-zhao-cui-generic-program-codex-reaudit-reply-2026-08-15.md`  
Verdict: `REVISE_BLOCKED_BEFORE_P1A_CONTENT_GATE`

## Executive verdict

Fable accepted the prior findings and repaired most of the mathematical
language in UB-1 and the program plan. Findings 2 through 6 are substantively
addressed as design-contract corrections: solver reuse is now a goal rather
than donor evidence, ties veto score claims, tau is explicitly viability-only,
the structural route is restricted to a global invertible subclass, and the
moving-point retained-density derivative is named.

The P1A claim-bearing gate still does not unblock. Two material closure defects
remain:

1. UB-1's V1 assembly is explicit, but the retention/evaluator equations in
   the same note still mix physical `x` and reference `z` notation. They do not
   state one measure-qualified returned object for `p_ret` and `q0_ret` at the
   point where the retained quadratic form is defined. This leaves the exact
   scalar contract vulnerable to the same double- or missing-conversion error
   the revision claims to have removed.
2. UB-2 row 2 cites `@TTFun/int_reference.m:1-40` for the exact normalizer of
   `h^2`, but that routine integrates the TT function `h`; it does not assemble
   the squared-TT mass. The pinned author implementation of the `h^2` mass is
   in `@TTSIRT/marginalise.m:25-51`, with the complete defensive normalizer at
   line 85. In addition, the master plan Section 10 still literally says
   “initial; anchors to be completed at UB-2,” contrary to the response's claim
   that it was synchronized.

These are bounded repairs, not a rejection of the research direction. P0 may
continue. P1A may be implemented only as diagnostic scaffolding until the two
closure defects are repaired and rechecked. P1B and P2 remain blocked by the
declared P1A prerequisites.

## Findings

### 1. F1 is not fully closed in the retained-object equations

UB-1 V1 now correctly declares:

```text
evaluate_reference_density(z): p_ret_ref(z)
evaluate_physical_density(x):  p_ret_phys(x)
```

and chooses `p_ret_ref` for the previous-state factor with only current-block
conversion. That part is correct and addresses the original target-assembly
ambiguity.

The same note then reintroduces ambiguous notation in the retained object and
its tangent:

- UB-1 lines 92-105 write `p_ret,t(x)` while the right-hand side is evaluated
  at `z = R^{-1}(x)`;
- lines 111-125 write `p_ret(z)` and `q0_ret` without the `_ref`/`_phys`
  qualifier;
- line 99 uses `q0_ret(x)` inside an otherwise reference-coordinate formula.

The V1 declaration does not cure an inconsistent definition of the object that
P1A is supposed to implement. A test cannot prove “both conventions” unless
the object payload, the defensive marginal, the normalizer measure, and every
evaluator output are each typed by convention.

Required bounded repair:

```text
p_ret_ref(z) = (H_L(z) E H_L(z)' + tau q0_ret_ref(z)) / Z_ref
p_ret_phys(x) = p_ret_ref(R^{-1}(x))
                * omega(R^{-1}(x)) / J_R(R^{-1}(x))
```

Then use `p_ret_ref` and `q0_ret_ref` consistently in the retained tangent and
V1 reference assembly. If a physical evaluator is also stored, state whether
`Z_phys` is numerically the same scalar under the corresponding measure
conversion or is a separately represented integral. `U-MEASURE-1` should assert
both evaluator identities and the two-step recursion, not merely mention them
in the test table.

Classification: `INSUFFICIENT` for the exact retained-object API; the V1
assembly choice itself is `correct` once these names are made binding.

### 2. The new UB-2 row-2 anchor does not support the claimed operation

The ledger row is:

```text
Exact Gram-chain normalizer of h^2
author anchor: @TTFun/int_reference.m:1-40
```

The cited routine starts with `figeqk = 1`, contracts each core with the
one-dimensional `integral`, and returns the integral of the TT represented by
the cores. It is a linear TT integral. It does not pair each core with itself
under a mass matrix and therefore does not establish the normalizer of `h^2`.

The pinned author `h^2` normalizer is implemented in
`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m`:

- lines 25-49 propagate the accumulated squared-mass factor through the
  remaining cores;
- lines 43-49 apply the one-dimensional mass operation and QR gauge;
- line 51 sets `obj.fun_z` to the squared mass;
- line 85 adds the defensive term to obtain `obj.z`.

The paper support is Zhao-Cui Section 3, Equation (13), and Proposition 2,
Equation (14), where the mass matrix is formed from the right accumulated core
and its integral. The local extracted paper is at
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:549-626`.

Replace row 2's primary author anchor with
`@TTSIRT/marginalise.m:25-51,85`. `@TTFun/int_reference.m:1-40` may remain as a
separate linear-TT integration reference only if the ledger labels that
different operation explicitly.

Classification: `wrong relative to the claimed source operation`; this keeps
the source-faithfulness gate open.

### 3. The master plan still contains the stale source-faithfulness table

The response says Plan Section 10 was synchronized to the revised ledger. The
working tree still has at
`docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md:500-509`:

```text
## 10. Source-classification route ledger (initial; anchors to be completed at UB-2)
```

It retains directory-level or incomplete anchors and does not carry the new
Lemma 1 transfer caveat, restricted structural scope, or exact operation-level
source table. This is not cosmetic: the binding Zhao-Cui policy blocks any
artifact using `source_faithful` without paper and author-source file/line
anchors. The master plan is itself a claim-bearing artifact and must not point
to a ledger state it does not contain.

Required repair: replace Section 10 with a pointer to UB-2 revision 2 plus a
short synchronized table whose `source_faithful` rows have exact paper and
author-code anchors. Mark the section `EXACT_ANCHORS_RECORDED`, or explicitly
state that UB-2 is the sole binding ledger and remove source-faithful claims
from the master plan.

Classification: `BLOCK_SOURCE_UNGROUNDED` remains active until synchronized.

### 4. Row-7's negative source claim needs narrower wording

The revised row 7 uses `@TTFun/grad_reference.m` to state that the author has a
TT evaluation gradient but no fit-through analytical score route. The positive
part is supported by the cited function: it differentiates the evaluated TT
with respect to reference inputs. One function is not, by itself, proof of the
absence of every fit-through score route in the entire author snapshot.

Use wording such as “no fit-through score route found in the inspected pinned
snapshot; `grad_reference.m:1-77` is an evaluation-gradient example” and record
the inspected source inventory/search basis. This is not a blocker once the
repository route is classified `extension_or_invention`, but it prevents an
overstrong negative source claim from being mistaken for a theorem about the
author code.

### 5. Later-phase repairs were not propagated to the phase specifications

The response says the P2A obligations and the UB-3 spatial-JVP requirement
were adopted. The master plan still has two weaker implementation descriptions:

- P2A at lines 364-367 is a “1-2 steps” prototype and lists runtime, peak
  bytes, and same-scalar FD, but does not bind the four solver-reuse checks
  adopted in UB-1. The existing pre-mortem risk that a short prototype passes
  while `T=120` tangent memory fails remains real.
- P2S at lines 377-385 says the score extension uses only `dot_S / dot_log_J`;
  it does not name the required `grad log p_ret . dot_S`, retained spatial JVP,
  inverse-map JVP, support status, and FD checks that Section 3.6 now calls
  load-bearing.

These do not independently block P1A, because P2A/P2S are later phases, but
they must be repaired before those phases can be admitted. A cost prototype
may be short for mode selection, but the plan must also bind a full-horizon
memory/retracing stress or explicitly classify that evidence as insufficient
for full-horizon feasibility.

## Per-artifact verdicts

| Artifact | Verdict | Reason |
|---|---|---|
| UB-1 revision 2 | `INSUFFICIENT` | V1 assembly, solver caveat, and tie veto are corrected, but retained/evaluator equations still do not carry one binding measure-qualified notation. |
| Retained quadratic-form contract | `AGREE` mathematically; `INSUFFICIENT` as an implementation API | The quadratic form and dual evaluator idea are sound; `p_ret`/`q0_ret`/normalizer measure ownership is not consistently written through the retention and tangent equations. |
| UB-2 revision 2 | `INSUFFICIENT` | Most anchors are improved, but row 2 cites a linear-TT integral for an `h^2` normalizer, and the master plan remains stale. |
| D1 tau policy | `AGREE` as a viability screen only | The response correctly preserves `viability_tuning_only` where no same-target reference exists. |
| D2 structural substitution | `AGREE` conditionally | The restricted global-diffeomorphism derivation and moving-point derivative are now correctly stated; P2S implementation text must still bind the full score terms. |

## Direct gate answer

The P1A claim-bearing gate remains `BLOCKED`.

The response's execution table remains correct in broad order:

- P0 may proceed with the dual-measure API and restricted structural scope;
- P1A may proceed only as diagnostic scaffolding;
- P1B and P2/P2A remain blocked by their declared prerequisites;
- P2S remains after UB-3 plus P2, with the full spatial-JVP contract;
- the density-kernel track remains independent of UB-3.

The next recheck should be bounded to:

1. UB-1 Sections 1(V1), 1(V5), and 2, verifying every retained-density and
   defensive term has an explicit measure suffix;
2. UB-2 row 2 and the exact-anchor status, including the corrected `marginalise`
   anchor;
3. master plan Section 10 synchronization;
4. master plan P2A/P2S implementation obligations.

No P1A implementation or experiment is authorized by this document. The
unblock decision is a content audit, not evidence that the named tests already
exist or pass.

## Pre-mortem

The repaired plan could still pass its stated gates and mislead if:

- V1 selects a reference evaluator but V5/tangent code implements the same
  object as a physical density, producing a two-step conversion error;
- `@TTFun/int_reference.m` is reused as if it were the author `h^2` mass and
  the resulting source claim survives because only the path name is checked;
- the master plan's stale source table is treated as historical prose while a
  reviewer or runtime manifest reads it as the active provenance contract;
- the tau sensitivity table is reported as a bias diagnostic despite the
  declared `viability_tuning_only` status;
- P2A passes at one or two steps but full-horizon tangent retention, retracing,
  or allocator behavior fails;
- P2S differentiates `S` and `J` but omits the retained-density spatial term;
- a negative “no author score route” statement is accepted from one evaluation
  gradient file without a complete inspected-snapshot inventory.

## Source-support boundary

This focused audit inspected the local Zhao-Cui paper text at the technical
anchors for Equation (13), Lemma 1, and Proposition 2/Equation (14), and the
pinned author files `@TTSIRT/eval_potential_reference.m`,
`@TTSIRT/marginalise.m`, `@TTFun/int_reference.m`, `@TTFun/cross.m`,
`@TTFun/build_basis_svd.m`, and `@TTFun/grad_reference.m`. Ch18b structural
pushforward assumptions and the repository's binding Zhao-Cui source-anchor
policy were also checked. This memo makes no citation-count, venue-ranking,
HMC-readiness, posterior-correctness, cost, or NAWM-feasibility claim.

## Final decision

`REVISE_BLOCKED_BEFORE_P1A_CONTENT_GATE`

The response is materially better and closes Findings 2-6 at the design level.
Repair the measure-qualified retained equations, correct UB-2 row 2's author
anchor, synchronize master-plan Section 10, and propagate the later-phase
obligations. A bounded content recheck can then decide the P1A gate without
reopening the program mission or phase ordering.

VERDICT: REVISE
