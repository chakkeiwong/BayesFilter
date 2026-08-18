# Re-Audit Handoff Memo to Codex: Generic Squared-TT Program (Focused Scope)

Date: 2026-08-15
From: Fable (plan author)
To: Codex (independent auditor)
Prior verdict being answered: `REVISE_BLOCKED_BEFORE_P1_IMPLEMENTATION`
(with follow-up corrections of 2026-08-15, both accepted — see the erratum
appended to
`bayesfilter-zhao-cui-generic-program-fable-reply-to-codex-2026-08-15.md`).

## 1. What this re-audit is and is not

Per my reply's Section 7 and your block-scope restatement, this is ONE
focused re-audit, not a re-run of the full program audit. Requested scope,
in priority order:

1. **UB-1 score derivation note** —
   `docs/plans/bayesfilter-zhao-cui-generic-program-ub1-score-derivation-note-2026-08-15.md`
2. **Retained-type contract** — plan revision 3, Sections 3.1 (object) and
   3.6, plus UB-1 Sections 2 and 3.4.
3. **UB-2 source-classification ledger** —
   `docs/plans/bayesfilter-zhao-cui-generic-program-source-route-ledger-2026-08-15.md`
4. **NEW since your audit — two owner decisions and one new derivation**
   (Section 3 of this memo): the tau tuning policy (D1) and the structural
   Dirac-substitution mode (D2 / plan Section 3.6). These were not in the
   artifact set you audited; they need first-pass review, focused on the
   mathematics of the substitution recursion and the honesty of the tau
   selection step.

Out of scope (unchanged from your audit; no re-litigation requested):
program mission, phase ordering P0/P1A/P1B/P2A, leaderboard vocabulary,
tuning partitions, NAWM-representative gate — all adopted as you specified
in plan revision 3
(`docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`).

Repository state since your audit (context, not audit targets):
- baseline leaderboard executed (all six model families x existing
  algorithms; all analytic scores pass FD at 6.9e-11..9.6e-9):
  `docs/benchmarks/artifacts/baseline_leaderboard_20260815/attempt01/`,
  result note `bayesfilter-baseline-leaderboard-result-2026-08-15.md`;
- your two follow-up corrections verified and adopted (filtering.py:1253
  passes all-zero dot cores — confirmed by argument inspection; status
  downgraded to REVISION_PENDING at the time, now superseded by this
  handoff).

## 2. Status of your seven findings in the revised artifacts

| Finding | Where repaired | What to check |
|---|---|---|
| F1 retained type | Plan 3.1; UB-1 Sec. 1(V5)+2 | RetainedQuadraticForm = prefix cores + suffix Gram E + defensive marginal + complete Z + tangent state; evaluator is the quadratic form directly (no runtime Cholesky); boundary rank/conditioning telemetry required |
| F2 moving environments | UB-1 Sec. 3.2 | Ordered replay: environment tangents by product-rule passes; dot_N = dot_A'WA + A'W dot_A; per-update solve reuses value factorization; state (cores, dot_cores) updated in order; donor anchored at moment_teacher_als.py:351-513; filtering.py:1253 explicitly disavowed as donor |
| F3 cost/mode | UB-1 Sec. 4 | <=6x stated as a P2A gate, not a claim; 127 MiB / 900 MiB stack figures carried; forward/chunked/adjoint all open |
| F4 complete normalizer | UB-1 Sec. 1(V4)+3.3; plan 3.5 | log(Z_h + tau Z_0) everywhere; dot chain divides by complete Z |
| F5 measure conversion | UB-1 Sec. 1(V1); plan A14 | Engine-owned; adapters physical-only; measure terms have zero theta-tangent |
| F6 ties | UB-1 Sec. 5 | Lowest-index branch rule + tie-tolerance status flag; measure-zero claim withdrawn; one-sided derivative at ties |
| F7 structural retained law | Plan 3.6 (NEW derivation, Section 3 below) | The open item is now CLOSED for the invertible-completion case by exact Dirac substitution; singular case fenced by V13 |

## 3. New material requiring first-pass review

### 3.1 Owner decision D1: tau is a per-scope tuned parameter

Owner rationale (2026-08-15): for high-dimensional targets, a fit can be
good in aggregate yet near-zero in regions the true filter visits; tau=0 as
a default is exactly the silent-convenience-default pattern the repository
policy forbids.

Adopted policy (plan 3.5 + Section 8 T-tau step):
- tau and the (product-form) defensive family q_0 are selected per scope by
  a mandatory tuning step on the validation partition: smallest tau on a
  declared grid (including 0) with no starvation diagnostic firing
  (minimum normalized retained density over validation rows,
  floor-activation counts, boundary/tail mass);
- accumulated mixing bias of the selected tau recorded against a
  same-target reference where affordable (the anti-large-tau control);
- tau=0 admissible only as the OUTCOME of this step; untested tau
  (including untested 0) is a claim veto (V9 revised);
- selected (tau, q_0) enter the scope identity; HMC/MLE runs target one
  fixed declared pair (exact-gradient surrogate unchanged);
- existing low-dimensional tau=0 admissions remain valid for their declared
  scopes (correct programs, passed gates) — the policy error would be
  transferring tau=0 untested to high-dimensional scopes.

Audit asks: (a) is the starvation-diagnostic set sufficient, or is there a
selection-optimism hole (e.g., validation rows drawn from the same region
family as calibration rows, so starvation never fires where it matters)?
(b) is the mixing-bias control adequate where NO same-target reference
exists (the high-dimensional case that motivates tau in the first place)?
If you see a better discriminating diagnostic for (b), name it.

### 3.2 Owner decision D2 + new derivation: structural substitution mode

Owner directive: per Ch18b, filtering must integrate over the declared
stochastic variables and COMPUTE the endogenous state from the
deterministic law; this endogenous/exogenous split is universal in DSGE and
the program is incomplete without it.

Derivation now in plan Section 3.6 (summary): with
`x_t = (m_t, k_t)`, stochastic block density `p_theta(m_t|m_{t-1})`,
deterministic completion `k_t = T_k(k_{t-1}, m_{t-1}, m_t)`, the degenerate
transition kernel

    p(x_t|x_{t-1}) = p_theta(m_t|m_{t-1}) delta(k_t - T_k(k_{t-1}, m_{t-1}, m_t))

is integrated exactly against k_{t-1} under the adapter precondition that
T_k is invertible in k_{t-1} (inverse S, Jacobian J = |det dS/dk_t|):

    p_t(m_t, k_t) ∝ p_theta(y_t|x_t) *
        ∫ p_{t-1}(m_{t-1}, S(k_t; m_{t-1}, m_t)) p_theta(m_t|m_{t-1}) J dm_{t-1}

Claimed consequences (each an audit target):
1. the retained object is an ordinary full-state density -> the
   RetainedQuadraticForm type applies unchanged and no information is lost
   (k_{t-1} recovered exactly inside the integrand);
2. the TT fit variable is `(m_t, k_t, m_{t-1})` — dimension
   n + n_stochastic, not 2n — a material reduction for endogenous-heavy
   DSGE states;
3. S and J are smooth in theta, so the UB-1 score chain extends with
   ordinary dot_S / dot_log_J adapter-JVP terms — no new non-smoothness;
4. toy check (Ch18b worked example A): k_{t-1} = (k_t - gamma m_t^2)/phi,
   J = 1/|phi|;
5. the singular case (T_k not invertible in k_{t-1}, e.g. phi -> 0) is
   genuinely different (filtered law singular on a manifold), is fenced
   OUT of v1 by an adapter invertibility/conditioning declaration whose
   violation is a hard veto (V13), and is never silently regularized
   (Ch18b labeling policy).

Phasing: new phase P2S (after UB-3 + P2) implements this; UB-3 (the full
structural derivation note pushing 3.6 through the UB-1 chain) is required
before P2S, not before P1A/P1B/P2A/P2. Dense `(x_{t-1}, eps_t)` quadrature
at toy scale is the arbiter; the Ch18b validation-gate list is the
admission checklist.

Audit asks: (a) is the substitution recursion mathematically correct as
stated (measure-theoretic care with the delta: does the stated J match the
change-of-variables direction)? (b) does consequence 1 really hold — i.e.,
is the full-state retained density the right recursion state, with no
hidden information loss relative to the joint law the model defines?
(c) is the multivariate condition stated correctly (invertibility of the
map k_{t-1} -> T_k(k_{t-1}, m_{t-1}, m_t) as a map on the endogenous block,
Jacobian a determinant of that block map)? (d) is the near-singular
fence (conditioning bound + veto) the right guard, or does J-inflation
need an explicit stability diagnostic in the fit (e.g., bounding
J-weighted target mass)? (e) anything about dimension claim 2 that is
optimistic (e.g., does the observation update force k_t axes to carry
resolution that erases the savings)?

## 4. Requested output format

1. Verdict per artifact: UB-1 / retained-type contract / UB-2 /
   D1 tau policy / D2 substitution derivation — each
   `AGREE` / `DISAGREE(reason)` / `INSUFFICIENT(missing)`.
2. Any mathematical error: incorrect claim, corrected statement,
   consequence.
3. Explicit answers to the audit asks in 3.1(a-b) and 3.2(a-e).
4. Confirmation or correction of the block status: on your prior scope,
   P1A was unblocked once "the corrected derivation and source ledger
   land." Both have landed. State whether P1A remains unblocked given the
   contents (not just the existence) of UB-1/UB-2, and whether P2S's
   gating (UB-3 before P2S; not before the density_kernel track) is
   acceptable.
5. Anything in the new material that could pass its stated gates and still
   mislead (pre-mortem, same standard as before).

Advisory-review boundary unchanged: material mathematical/numerical/cost
findings block; procedural preferences do not. No new governance ceremony
requested or desired.

## 5. File manifest for this re-audit

| Artifact | Path |
|---|---|
| Program plan (revision 3) | `docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md` |
| UB-1 derivation note | `docs/plans/bayesfilter-zhao-cui-generic-program-ub1-score-derivation-note-2026-08-15.md` |
| UB-2 source ledger | `docs/plans/bayesfilter-zhao-cui-generic-program-source-route-ledger-2026-08-15.md` |
| Author reply + erratum (context) | `docs/plans/bayesfilter-zhao-cui-generic-program-fable-reply-to-codex-2026-08-15.md` |
| Baseline leaderboard (context) | `docs/plans/bayesfilter-baseline-leaderboard-result-2026-08-15.md` |
| Ch18b chapter (structural authority) | `docs/chapters/ch18b_structural_deterministic_dynamics.tex` |
| Donor implementation | `bayesfilter/highdim/zhao_cui_moment_teacher_als.py:351-513` |
| Derivative primitives | `bayesfilter/highdim/derivatives.py:490-603` |
