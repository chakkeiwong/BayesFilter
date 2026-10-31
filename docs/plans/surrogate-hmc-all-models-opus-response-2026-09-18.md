# Response to the Opus verdict on the surrogate-HMC master

2026-09-18. Question: do we agree with the
[Opus verdict](surrogate-hmc-all-models-opus-verdict-2026-09-18.md) on the
[master program](surrogate-hmc-all-models-master-program-2026-09-18.md)?

**Partly. I agree with its favorable assessment of the research design and with
several useful clarifications. I reject F1: its claimed requirement that the
force and acceptance endpoint share the same frozen particle bank and controls
is mathematically wrong. That proposed restriction must not be implemented as
a condition of surrogate-HMC validity.** F2 is useful reporting clarification,
already substantially covered by the master's explicit precision distinction.
The review does not establish a new mathematical defect in the program.

This response preserves both the master and the Opus verdict unchanged. It is
a document/source review with a local derivation, not a new experiment or an
independent numerical validation. No GPU job, sampler, or reviewer was launched.

## F1 confuses a fixed acceptance density with a matching force

Let the endpoint and force have separate fixed identities:

```text
U_v(z) = U_D(z; omega_v, c_v)
g_f(z) = g_D(z; omega_f, c_f).
```

Both are deterministic functions of position after their respective inputs
are frozen. The force must be finite on the qualified proposal domain, or have
the explicitly qualified finite extension described in the master. There is
no assumption that `omega_f = omega_v`, `c_f = c_v`, or `g_f = grad U_v`.

For fixed positive mass, step size and leapfrog count, the kicks and drifts
built from `g_f` are volume-preserving shears. Their symmetric composition with
momentum reversal gives an involution `T_f`: `T_f(T_f(x)) = x`, where
`x=(z,p)`. This inverse uses the same `g_f`; it does not use `U_v` at all.

Define the extended target and acceptance by

```text
rho_v(x) proportional to exp(-U_v(z) - K(p))
alpha(x) = min(1, rho_v(T_f(x)) / rho_v(x)).
```

Then, with the volume-preserving change of variables,

```text
rho_v(x) alpha(x)
  = min(rho_v(x), rho_v(T_f(x)))
  = rho_v(T_f(x)) alpha(T_f(x)).
```

This is detailed balance for the paired extended states. Momentum refresh from
the matching Gaussian preserves the same extended target, and its position
marginal is the endpoint posterior. The proof makes no identification between
the force's frozen inputs and the endpoint's frozen inputs. It establishes
ideal-arithmetic invariance, not convergence or numerical correctness.

A direct counterexample to F1's claimed necessity is `U_v(z)=z^2/2`, unit mass,
and `g_f(z)=0`. One leapfrog step with momentum reversal gives
`(z',p')=(z+h*p,-p)`. Since `K(p')=K(p)`, acceptance is
`min(1, exp(-(z'^2-z^2)/2))`. With fresh standard-normal momentum this is
ordinary symmetric Gaussian random-walk Metropolis for the standard-normal
target. The zero force has no particle-bank relationship to that endpoint,
yet the target is invariant. This counterexample demonstrates the mathematical
distinction; it does not nominate zero force as the research method.

Opus's valid concern would be using `U_D(z;omega_1)` for the current state and
`U_D(z';omega_2)` for the proposed state without a valid augmented-state method.
That changes the acceptance target. However, a different frozen bank for the
**force** does not imply different banks for the two **endpoint evaluations**.
The inference made in F1 does not follow.

The corrected consumer requirement is:

> Every current, cached, proposed and replayed endpoint evaluation uses the
> same declared endpoint binding `(omega_v,c_v)` throughout retained sampling.
> Every forward and reverse force evaluation uses its own declared frozen
> force binding `(omega_f,c_f)`. Verify each binding independently and prevent
> force evaluation from mutating endpoint state or its cache. The two bindings
> may differ.

Sharing inputs is a permissible experimental choice and can be useful for
attributing force errors. It is not a detailed-balance requirement. Freezing
only within an individual trajectory is also insufficient for this program's
claim to sample one fixed finite-DCTR posterior throughout the retained chain.
The master already requires the retained endpoint and force to remain frozen;
explicitly separating their binding checks would make that requirement clearer.

Useful future wiring tests should include different fixed force/endpoint
identities, deterministic replay, and detection of endpoint/cache mutation.
They should not assert equality of the two identities. Existing numerical
failure, support and floating-point qualifications remain necessary.

## Disposition of the other findings

| Finding | Assessment | Correct treatment |
|---|---|---|
| F2: mixed precision | Agree with adding a positive name and structured precision fields; disagree that a label supplies missing mathematical validity. Section 9 already distinguishes FP64 integration from GPU FP32/TF32 DCTR and says an all-FP32 sampler is a separate scope. | Record integrator, momentum, force, endpoint and reference dtypes separately. B has an FP64 Kalman endpoint; C has a promoted FP32 DCTR endpoint. A single label such as `FP64_INTEGRATOR_FP32_DCTR` must not obscure this arm difference. Keep the existing separate binding/parity requirements. |
| F3: noise-scale priors | Agree that the proposed width and scale need explicit provenance, prior-predictive checks and limits on interpretation. Arbitrary source scales do not make a deliberately synthetic sampler benchmark invalid or circular by themselves. | A proper, frozen benchmark prior can define a valid test target even when it is not empirically calibrated. Require a defensible interpretation for real-model inference. Any data-derived prior must disclose the construction and avoid undeclared use of held-out answers. `SCOPE_UNRESOLVED` applies when the target or intended claim cannot be justified, not automatically when a fixture scale is convenient. |
| F4: measured budget | Agree; the master already requires measurements and does not assert that all scopes fit. | Measure complete costs before launch. Fewer arms or looks can define a documented partial test, but cannot silently reduce the requirements for a complete result. Preserve required scopes and report the remainder as incomplete or under-budgeted. |

The review's broad endorsement of the uncertainty design is conditional on
reference qualification, stationarity, adequate mixing and reliable
autocorrelation-aware error estimates. A two-look allocation does not make those
assumptions true or turn empirical reference refinement into a rigorous bound.
The master states these limits; the review is design-level agreement only.

The review also says the premise would collapse if a current artifact issuer
cannot represent an imperfect proposal field. That is too strong. Such a
finding would block the particular consumer until its interface is repaired;
it would not refute the surrogate-HMC argument above. Preserve canonical
algorithm identity and exact-score admission rules while giving this different
scientific use an honest classification. The public HMC capability already
advertises a typed endpoint-corrected position-field route; full DCTR integration
remains to be verified.

## Source evidence and limits

Inspected [master section 3](surrogate-hmc-all-models-master-program-2026-09-18.md),
including independent controls, the involution proof and cached endpoint; its
mixed-precision and consumer checks in section 9; and the public
[HMC tuning interface](../reference/hmc-tuning-interface.md).

In [neural_force_hmc.py](../../bayesfilter/inference/neural_force_hmc.py),
`neural_force_hmc_transition` at line 603 accepts separate force and target
bindings. The proposal receives the force; proposed potential evaluation
receives the target; the full energy difference uses that potential and the
cached current potential. This supports the architectural distinction. The
cache's consistency with its binding remains a consumer-level requirement;
reading the function does not verify a future all-model consumer.

[test_neural_force_hmc.py](../../tests/test_neural_force_hmc.py), line 379,
contains a test of a scaled Gaussian force against a different Gaussian
endpoint. It was inspected, not run for this response. It does not itself
test separate DCTR particle-bank bindings or establish all-model validity.

[tuning_contract.py](../../bayesfilter/inference/tuning_contract.py), line 850,
advertises public `tune_hmc_kernel` support for a typed endpoint-corrected
position-field route. This source check addresses one item Opus explicitly
left unchecked; it does not close the full DCTR issuer or consumer work.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Accept useful clarifications; reject F1's required identity constraint | The research design tests imperfect-force surrogate HMC with separately defined targets | F1 is disproved as a mathematical necessity; no new design blocker established by this verdict | Actual consumer wiring, qualified references, complete coverage and affordable precision | Clarify independent binding checks and precision reporting; proceed through the existing implementation prerequisites when execution is requested | Implemented runner correctness, convergence, all-model success or production readiness |
