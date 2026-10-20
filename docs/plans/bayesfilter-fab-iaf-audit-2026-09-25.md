# FAB, IAF and importance-sampling chapter audit

The new chapter is `docs/chapters/ch26d_modern_importance_sampling.tex`, included
by `docs/main.tex`. Its mathematical argument was checked against the locally
preserved papers and the pinned author FAB implementation. The audit supports
the stated identities under their stated assumptions. It does not prove that
the q20 auxiliary density has a finite normalizer, that a finite AIS batch
finds every important region, or that a trained map is ready for posterior HMC.

## Sources and implementation correspondence

Author FAB repository: `lollcat/fab-jax`, revision
`c9f991366ca94b2678a7ed620bc9e12655cfef1d`, preserved under
`.localresources/fab-jax-c9f9913`. Its MIT notice is retained in the derived
TensorFlow module. The author repository was read; its dependencies were not
installed and its JAX implementation was not executed.

| Mechanism | Source anchor | Disposition |
|---|---|---|
| Alpha-2 bridge | `fabjax/sampling/base.py:75`; FAB §3.1, Eqs. 4–7 | Same log-density and spatial-score coefficients; alpha fixed at 2 |
| AIS ordering | `fabjax/sampling/smc.py:78` and `:122` | Same initial increment, interior mutation, next increment, and omission of a terminal mutation |
| HMC | `fabjax/sampling/mcmc/hmc.py:20`, `blackjax_hmc_rewrite.py` | Identity-mass leapfrog, MH correction, adaptation after each mutation; stricter path validity checks |
| Gaussian Metropolis | `fabjax/sampling/mcmc/metropolis.py:17` | Same symmetric proposal and MH ratio; scale held fixed within each temperature and adapted once afterward |
| Detached fresh loss | `fabjax/train/fab_without_buffer.py:33`; paper Eq. 7 | Deliberate paper-based scaling: weighted sum; author JAX adds a factor 1/batch by taking a mean |
| Replay correction | `fabjax/train/fab_with_buffer.py:20`; paper §3.2 and Algorithm 1 | Detached old/current density ratio; capped loss correction and uncapped priority update |
| Replay initialization | `fabjax/train/fab_with_buffer.py:122` | Corrected to `floor(minimum/batch)+1` passes. Configured 40-batch minimum means 41 fill passes |
| Replay sampling | `fabjax/buffer/prioritised_buffer.py:10` and `:82` | Gumbel top-k without replacement and shuffle; distinct indices across the update minibatches |
| Optimizer/sampling order | `fabjax/train/fab_with_buffer.py::step` | Fresh AIS uses pre-update map, old buffer supplies updates, then fresh data enter the buffer; local scheduling is serial |
| IAF density and inverse | `neutra_transport.py`, `neutra_transport_core.py`; Hoffman §4.1.1 and author code cited in `neutra-implementation.md` | Existing canonical map preserved; FAB introduces no alternate architecture |
| Frozen consumer binding | `run_q20_configured_hmc_2026_09_24.py:105` | Export binds to the beta-1 adapter signature used by the existing consumer; underlying target signature recorded separately |

The parameterized components are ported, not a bitwise JAX execution. TensorFlow
stateless random streams differ from JAX streams. TensorFlow/Keras Adam retains
the local optimizer convention, including its epsilon placement; it is not an
Optax optimizer reproduction. The target callback remains the existing native
batched FP64 value/score/status computation. The FP32/TF32 map is evaluated in
FP64 for the final represented-map diagnostics. The Metropolis port does not
need the flow's spatial derivative but retains the target callback's supplied
score/status check. Initial invalid draws veto an attempt rather than being
replaced, a deliberate stricter departure from the author code.

The initial implementation began replay one batch too early. The first three
exploratory workers were interrupted before their first optimizer update,
the boundary was repaired, the regression passed, and their valid initialization
prefixes were resumed into new output directories. No trained result from the
earlier boundary is used. The experiment plan and interruption notes preserve
the exact costs and counters.

The first real GPU replay call exposed another integration gap:
`StatelessShuffle` was unsupported by XLA_GPU. Initial compiled tests had
covered AIS and loss updates separately but had not compiled replay selection.
The repair uses a permutation induced by independent FP64 random keys and
adds a compiled replay regression. A complete GPU/XLA smoke now executes
initialization, replay selection, two actual optimizer updates and checkpoint
round-trip successfully. This closes that engineering failure; the q20 fits
remain untrained at the wall-clock deadline.

## Mathematical checks and their limits

| Claim | Checked argument | Remaining condition or limitation |
|---|---|---|
| Importance-weight variance | Substitute `W=gamma/q`, divide by `Z²`, and subtract one | Support and finite second moments are necessary; sample ESS cannot detect an unvisited mode |
| SNIS variance | First-order expansion of the ratio gives `W(f-mu)/Z` | CLT and second-moment assumptions; finite-sample ratio bias remains |
| Product dimension identity | Tonelli factors the nonnegative `p²/q` integral | Factorization is an explicit assumption, not a generic high-dimensional guarantee |
| AIS weighted expectation | Induction: weighting changes the marginal to the next unnormalized density; an invariant fixed kernel preserves it | Same-batch scale adaptation is not covered by this fixed-kernel proof |
| FAB gradient | Differentiate `gamma²/q`, factor out `grad log q`, and divide by `J` | Manuscript states continuous differentiability, finite `J`, and a dominating differentiated integrand; these are not proved globally for the q20 map |
| Replay correction | Ratio of the two unnormalized auxiliary targets is `q_old/q_current`; successive ratios telescope | Self-normalization, capped correction, sampling without replacement and stale priorities prevent an exact unbiased-gradient claim |
| Gaussian tail condition | Exponent is `-x²(1/sigma_p²-1/(2 sigma_q²))` | Integral finite iff `sigma_q² > sigma_p²/2`; finite ray checks cannot prove the analogous global condition for a neural map |
| IAF inverse derivative | Differentiate `T_phi(z)=x` at fixed `x`; include explicit and inverse-latent dependence in `log q_phi(x)` | Numerical finite differences test selected directions, not a formal proof of the complete program |
| Frozen NeuTra posterior | Change of variables with the full log determinant preserves the declared posterior | Mixing, coverage, numerical validity and posterior uncertainty require downstream evidence |

The AFT/CRAFT discussion separates exact change-of-variables weights from
adaptive-map asymptotics. The defensive-mixture bound states the substantive
domination assumption. Kernel rates are fixed-dimensional smooth-density
calculations, not dimension-free adaptive guarantees. SCLD Proposition 2.3
concerns a KL-estimator relative-error bound; its log-variance objective is
instead anchored to §2.3 and Theorem A.2. This source distinction was repaired
in the text. Annealing Flow's inspected author repository contained only a
README at the inspected revision; no code reproduction is claimed for it.

## MathDevMCP execution

Used the installed MathDevMCP CLI in the `mathdevmcp-backends` environment,
with `PYTHONPATH=/home/ubuntu/python/MathDevMCP/src`. This calls the installed
package's audit functionality even though no MathDevMCP connector tools were
exposed in this session. Exact commands and source hashes are preserved under
`docs/plans/artifacts/neutra-fab-2026-09-25/`.

Seven bounded symbolic checks returned `equivalent`: bridge coefficients,
AIS increment, replay ratio, replay telescoping, density-derivative
factorization, Gaussian tail exponent, and SNIS linearization. Their inputs
are algebraic formalizations; nonzero denominators and the manuscript's
probability assumptions remain required.

The whole-document rigor command failed inside MathDevMCP with
`KeyError: 'evidence_refs'` in `document_exposition.py:621`. Its failure is
preserved in `mathdev-initial-failure.json`; the audit was narrowed to supported
bounded operations rather than treating that crash as a mathematical finding.

Final source-bound label audits for the FAB gradient, replay ratio and AIS
identity completed but returned `unverified`. They reported missing formalized
regularity constraints, notation requiring manual formalization, and an unsafe
multiline extraction. No mismatch was reported. The manuscript explicitly
states the differentiation assumptions, and the manual arguments above address
the substantive claims, but this does not convert the tool's statuses into
formal certificates. The raw `*-final-audit.json` results retain those limits.

## Engineering and rendered-document checks

The final suite passes all sixteen focused CPU-hidden reference/mechanics
checks in 72.01 seconds. The complete GPU/XLA replay smoke passes separately
in 15.97 seconds, including two optimizer updates and checkpoint round-trip.
They cover bridge/replay algebra, detached gradients, inverse parameter
derivatives, Gaussian weights and moments, distinct replay sampling, JSON
checkpoint continuation, invalid-target rejection, correction clipping versus
priority refresh, XLA update parity and both mutation operators. The CPU runs
are explicitly reference exceptions and do not establish GPU training quality.

The complete monograph compiles with all citations and references resolved.
The new chapter's rendered pages were inspected. A crowded running header and
a long inline responsibility formula were repaired; no layout warning remains
inside this chapter in the final build log. Unrelated pre-existing monograph
layout warnings remain outside the chapter. Build hashes, command and output
are in `document-build-manifest.json`. Human review of the chapter's prose
remains pending; compilation and model inspection do not certify a human voice.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the TensorFlow FAB trainer for bounded exploratory execution | Focused mechanics and source correspondence checked | No known weight/derivative mismatch after the initialization repair | Target-specific exploration, integrability and training adequacy | Complete the seeded fits and mandatory probes under the existing cap | Production training quality, posterior readiness or statistical superiority |
| Retain the chapter as a compiled provisional manuscript | Source-grounded derivations, explicit assumptions and readable rendering | No unresolved citation or known mathematical mismatch | MathDevMCP could not certify the general integral arguments; human prose review pending | Preserve the audit limitations and obtain ordinary manuscript review | Formal verification of the whole chapter or of q20 tail conditions |

Post-audit red team: the strongest remaining concern is that an apparently
healthy finite training pass can coexist with unvisited regions or a divergent
alpha-2 auxiliary integral. A finite geometry improvement does not answer
those questions. Evidence of such a defect would overturn a training-validity
claim; poor finite-map geometry instead rejects the candidate or its training
budget without rejecting FAB as a research direction.
