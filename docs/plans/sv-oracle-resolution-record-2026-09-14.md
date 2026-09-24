# SV Oracle Resolution Record: KSC-SV, Generalized SV, Actual SV

**Date consolidated:** 2026-09-14
**Status:** RESOLVED — preserved here because the resolution was rediscovered
several times, each time expensively
**Sources:** `bayesfilter-ledh-canonical-rebuild-execution-plan-2026-08-21.md`
(ledger entries 2026-08-23, 2026-08-24), `bayesfilter-actual-sv-canonical-transformed-likelihood-comparison-plan-2026-08-11.md`

---

## Why this record exists

The oracle relationship between the SV variants was root-caused and fixed on
2026-08-23, then re-audited on 2026-08-24. Before that it was misdiagnosed at
least twice, and the misdiagnoses were *plausible enough to act on* — one of them
produced a "needs a reviewed manifold-aware extension" verdict that was wrong
about the cause and would have sent work in the wrong direction.

The resolution currently lives inside a long execution ledger for a different
program. On 2026-09-14 an agent planning multi-model SQMC work read
`ksc_sv_canonical_model`'s docstring, saw the earlier "not exact for native SV"
language from the superseded v2 program, and recorded the generalized-SV /
KSC-SV oracle as an **unresolved blocker** — the same wheel, about to be
reinvented. Hence this standalone record.

---

## The mathematical relationship (the thing that keeps getting lost)

**Actual SV** observes the raw return \(y_t\).
**KSC-SV** observes the transformed \(z_t = \log(y_t^2)\).

The transform is a **bijection of \(|y_t|\)** and it is **\(\theta\)-independent**.
Therefore, with the same observation-density family:

- **Value** differs by exactly the \(\theta\)-independent Jacobian constant
  \[
  \sum_t \log\left|\frac{d(\log y_t^2)}{dy_t}\right| = \sum_t \log\frac{2}{|y_t|}
  \]
- **Score** with respect to \(\theta\) is **IDENTICAL** — zero difference, not
  "small". No \(\theta\) path touches the Jacobian, so it contributes nothing to
  \(\nabla_\theta\).

The canonical transformed observation density is exact:
\[
r_t(x_t;\theta) = z_t - 2\log\beta(\theta) - x_t, \qquad
p_\theta(z_t \mid x_t) = \frac{1}{\sqrt{2\pi}}
  \exp\!\left(\tfrac12 r_t - \tfrac12 e^{r_t}\right)
\]

**Consequence for oracles.** KSC-SV has a **dense Kalman filter oracle** (the
moment-matched Gaussian of the KSC log-chi-square mixture gives a linear-Gaussian
system in the transformed coordinate). Because the SV variants are related by a
\(\theta\)-independent bijection, that oracle transfers:

- a **generalized-SV / actual-SV score oracle** is obtained from the KSC-SV dense
  Kalman oracle **directly**, with no correction, because the score is invariant;
- the corresponding **value** oracle requires adding the Jacobian constant
  \(\sum_t \log(2/|y_t|)\), which is computable in closed form from the data
  alone.

So a 19000-nat value discrepancy between the families is **impossible** for a
faithful onboarding — that magnitude was the signal that found the real defect.

---

## Failure history (why the naive reading is wrong)

### Misdiagnosis 1 — "near-singular Q needs a manifold-aware extension"

**Observed:** KSC canonical value wildly wrong (~19000 nats off) while the
analytical score stayed self-consistent at 1.5e-10.

**Concluded (WRONGLY):** the value is not comparable until near-singular-\(Q\)
handling — a manifold-aware flow or exact-constraint transition — is added. The
KSC row was flagged NOT COMPARABLE pending a reviewed extension.

**Actual status:** *wrong relative to the true cause.* Recorded as **SUPERSEDED**
in the 2026-08-23 ledger entry. No extension was needed. The state space was wrong.

### Root cause (2026-08-23, found from an owner challenge)

The onboarding had promoted `log_beta` — a **parameter** in the reference target,
where state dimension is **1** — into a second **state** dimension with a
fabricated near-deterministic transition (\(Q_{22} = 10^{-8}\)). The flow's
legitimate displacement in that invented dimension was then priced at \(1/10^{-8}\)
by the fabricated density, producing the enormous value error.

**Fix:** re-onboard KSC with the correct **1-D latent state** (log-volatility
AR(1) \(h' = \gamma h + \eta\)) and `log_beta` as an **observation-offset
parameter** entering as \(2\log\beta\).

Corrected agreement: canonical −6.75 / bootstrap −6.90 / UKF −6.82, score
self-consistency 3.9e-12.

### Misdiagnosis 2 — "cross-algorithm agreement validates the model"

**The deeper lesson, recorded as an E-class recurrence.** Onboarding oracle gates
verify **self-consistency of whatever model was defined**, not **fidelity to the
reference model definition**. Worse, the slice-2 bootstrap and UKF comparators
*consumed the same model objects*, so cross-algorithm agreement **cannot detect
shared model infidelity**. Three independent algorithms agreeing on a wrong model
still agree.

Only **independent-reference density-equality gates** catch this class.

### Follow-on audit (2026-08-24) found two more of the same class

1. **diagonal-LGSSM** used `eye(3)` where the frozen target's `_LGSSM_MATRIX` is a
   specific non-identity matrix — fixed with reference constants.
2. **generalized-SV** used fixed observation variance 1 where the reference
   `NativeGeneralizedSVSSM` is **heteroskedastic** \(N(\beta s, e^{h})\) — fixed
   via an optional non-Gaussian observation-density surface in
   `NonlinearScoreModel` (density plus analytical tangent; the Gaussian remains
   the flow's proposal input).

Austria SIR and predator-prey were audited faithful.

---

## Live gates (verified present 2026-09-14)

| Gate | Location | What it binds |
|---|---|---|
| `test_ksc_equals_actual_sv_up_to_constant` | `tests/highdim/test_ledh_canonical_models.py:346` | Jacobian-constant value relation + score invariance + value-scale sanity. Builds the actual-SV lane **independently from raw \(y\) inside the test**, not from the KSC factory. |
| `test_ledh_canonical_model_fidelity.py` (4 gates) | `tests/highdim/` | gen-SV densities vs native reference at 1e-10; LGSSM matrix constants; Austria variance/extraction structure; predator-prey noise scales |
| meta-governance registration | `tests/highdim/test_ledh_canonical_meta_governance.py:87` | registers the equivalence gate under `invariance_or_reduction` so it cannot be silently dropped |

---

## Correct reading of the KSC docstring

`ksc_sv_canonical_model`'s docstring says the flow input is a *moment-matched
Gaussian* of the KSC log-chi-square mixture. Two distinct claims must not be
conflated:

- **The observation model is a mixture approximation of the log-chi-square law.**
  This is the KSC construction itself and is inherent to the KSC target — it is
  what makes the transformed system linear-Gaussian and hence Kalman-filterable.
- **The dense Kalman filter is exact FOR THAT KSC target.** Given the KSC mixture
  observation model, the dense Kalman recursion is the exact filter.

The earlier v2-program statement "not exact for native SV" was about approximating
**native SV** by the KSC *mixture* — a real caveat about the KSC construction —
**not** a statement that the SV variants lack an oracle relationship. Reading the
docstring as "no oracle available for SV" is the error this record exists to
prevent.

---

## What this unblocks

The SQMC master program (`sqmc-master-program-2026-09-12.md`) recorded the
generalized-SV / KSC-SV oracle as an unresolved blocker for extending the tuned
4-route comparison beyond LGSSM. **That entry is wrong and is superseded by this
record.** For the SV family:

- **score oracle:** KSC-SV dense Kalman, transferred unchanged (score invariance);
- **value oracle:** the same, plus the closed-form Jacobian constant
  \(\sum_t \log(2/|y_t|)\).

Remaining models still lacking an exact reference: **Austria SIR** (9-D nonlinear
RK4) and **predator-prey** (2-D nonlinear RK4). The reference problem is real for
those two; it is **not** real for the SV family.

Conditions to carry when using this oracle:
1. Use the **1-D** latent state with `log_beta` as an observation-offset
   parameter. The 2-D state formulation is the known defect.
2. Keep `test_ksc_equals_actual_sv_up_to_constant` green; it is the executable
   statement of the relationship.
3. Report value comparisons **with** the Jacobian constant stated explicitly, and
   score comparisons as requiring **zero** offset.
4. Do not claim the KSC mixture equals native SV. The oracle is exact for the KSC
   target; the mixture-vs-native gap is a separate, documented approximation.

---

## Method lessons worth keeping

1. **A \(\theta\)-independent reparameterization cannot change the score.** When
   two lanes that differ only by such a transform disagree in score, the
   onboarding is wrong — this is a hard check, not a tolerance question.
2. **An impossible magnitude is a gift.** The 19000-nat gap was diagnostic
   precisely because no faithful onboarding could produce it. Reach for
   "what would make this impossible?" before "what tolerance accommodates this?"
3. **Self-consistency is not fidelity.** A score matching its own program's JVP at
   1e-12 says nothing about whether the program is the intended model.
4. **Shared inputs void cross-algorithm agreement.** If comparators consume the
   same model object, their agreement carries no fidelity information. Only an
   independent reference definition does.
5. **A superseded diagnosis must be marked superseded.** "Manifold-aware extension
   needed" survived in notes after being disproven; that is how a wrong lead gets
   re-followed.

---

## Changelog

- 2026-08-11: transformed-likelihood comparison plan establishes the exact
  transformed density and \(\theta\)-independence of the Jacobian
- 2026-08-23: KSC-SV defect root-caused (invented 2nd state dimension) and fixed;
  equivalence gate added; "manifold-aware extension" diagnosis superseded
- 2026-08-24: fidelity audit finds two more infidelities of the same class
  (LGSSM identity matrix, gen-SV homoskedastic observation); fidelity gate class
  added
- 2026-09-14: consolidated into this standalone record after the resolution was
  again treated as an open blocker in the SQMC master program
