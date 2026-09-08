# MathDevMCP audit: adaptive-replay NeuTra mathematics

Date: 2026-08-21
Status: `PARTIAL_SYMBOLIC_CERTIFICATION_HUMAN_REVIEW_REQUIRED`

Primary note:
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md`

## Audit verdict

MathDevMCP certified the pointwise deterministic-mixture denominator
cancellation used inside Theorem 1. It did not certify the measure-theoretic
importance identity, either stochastic-approximation theorem, the KL equality
condition, the Rosenblatt existence result, or the ideal-HMC argument. Those
obligations exceed the bounded symbolic routes exercised here and require
mathematical review.

No MathDevMCP query produced a counterexample to the note. That is not a proof
of the full note. The correct disposition is:

```text
MATHDEVMCP_AUDIT: PARTIAL
DERIVATION_DISPOSITION: CONDITIONAL_NOTE_READY_FOR_INDEPENDENT_HUMAN_REVIEW
FINITE_SSL_LSTM_SUCCESS_CLAIM: UNSUPPORTED
```

## Environment

`mathdevmcp doctor` returned `ok: true` under Python 3.13.13 in the `tfgpu`
environment. SymPy 1.14.0, SageMath 10.7, LaTeXML 0.8.8, and Pandoc 3.10 were
available. Lean had no configured default toolchain, and LeanDojo was not
available in the active environment. Therefore this audit has no Lean proof
certificate.

## Check ledger

| Obligation | MathDevMCP result | Evidence class | Disposition |
|---|---|---|---|
| Mixture cancellation `(sum alpha r) * p/(sum alpha r) = p` off the zero set | `proved`; SymPy simplified the difference to zero | `backend_certificate` | Certified only as pointwise scalar algebra under a nonzero denominator |
| Integral linearity and interchange in Theorem 1 | `gap_found`, `not_encodable`; parser treated `integral` as an undefined callable | `human_review_required` | The written conditional-expectation proof remains manual; support and integrability assumptions are explicit |
| Completeness of replay stochastic-approximation assumptions | `inconclusive`; bounded rule set found no route | `human_review_required` | Not certified; the note was tightened to a bounded Lipschitz Poisson solution plus strong mean-gradient stability |
| Completeness of adaptive fresh-block plus summable-replay assumptions | `inconclusive`; bounded rule set found no route | `human_review_required` | Theorem 2A remains a manual martingale/summable-perturbation proof for Fable to audit |
| Completeness of common-minimizer assumptions | `inconclusive` | `human_review_required` | KL nonnegativity and equality are proved manually in the note |
| Hybrid zero arithmetic | `check-proof-obligation` returned `equivalent`; `prove-or-counterexample` later returned `inconclusive` after its router misclassified the expression as needing matrix/domain review | conflicting tool routes | Treat only the trivial arithmetic as manually obvious; do not promote either route to certification of Theorem 3 |
| Ideal-HMC endpoint arithmetic `z*0+p*1=p` | `equivalent` | `backend_verified` | Certifies the last scalar simplification only, not Hamiltonian dynamics or independence |
| Direct trigonometric form at `pi/2` | `unverified` because the parser did not normalize the trigonometric expression | diagnostic only | The note's elementary derivation remains manual |
| Current weighted loss against `neutra_weighted_training.py` | `scope_limited_match`; all ten supplied code terms matched | structural supporting evidence | Supports the description of the implemented fixed empirical loss; not semantic correctness proof |
| Claim that the current 600 SMC rows and finite dense IAF are proved to train successfully and globally mix | `unsupported` | claim-boundary evidence | Correctly retained as an explicit nonclaim |
| Final review packet `ssl-lstm-q20-adaptive-replay-neutra-math-r3-20260821` | `review_ready` | handoff metadata | Packet readiness is not proof; independent review remains required |

## Exact command record

The executable was
`/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp` in every command below.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp doctor

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp check-proof-obligation \
  "(a1*r1+a2*r2)*(p/(a1*r1+a2*r2))" "p" \
  --assumption "a1*r1+a2*r2 != 0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp prove-or-counterexample \
  "(a1*r1+a2*r2)*(p/(a1*r1+a2*r2)) = p" \
  --assumption "a1*r1+a2*r2 != 0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp derive-step \
  "(a1*r1+a2*r2)*(p/(a1*r1+a2*r2))" "p" \
  --assumption "a1*r1+a2*r2 != 0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp debug-derivation \
  --step "a1*integral(r1*p*f/m)+a2*integral(r2*p*f/m)" \
  --step "integral((a1*r1+a2*r2)*p*f/m)" \
  --step "integral(p*f)" \
  --assumption "m=a1*r1+a2*r2" \
  --assumption "m is nonzero on target support" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp assumptions-for \
  "fixed-capacity replay stochastic approximation converges to phi_star under content-independent geometric refresh, a bounded Lipschitz Poisson solution, Robbins-Monro steps, and strong mean-gradient stability" \
  --provided-assumption "fresh blocks have the correct unnormalized forward-gradient mean" \
  --provided-assumption "fresh reverse gradients are conditionally unbiased with bounded second moment" \
  --provided-assumption "the block-gradient Poisson equation has a bounded Lipschitz solution" \
  --provided-assumption "the hybrid gradient is strongly stable around a unique phi_star" \
  --provided-assumption "Robbins-Monro step sizes are nonincreasing"

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp assumptions-for \
  "the positive hybrid forward and reverse KL objective has the exact transport as its common global minimizer" \
  --provided-assumption "a > 0" --provided-assumption "b > 0" \
  --provided-assumption "Z > 0" \
  --provided-assumption "the flow family contains q_phi equal to pi"

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp assumptions-for \
  "adaptive fresh-block stochastic gradient with a bounded stale-replay perturbation converges to phi_star" \
  --provided-assumption "the proposal is selected from past history and frozen before each fresh block draw" \
  --provided-assumption "the fresh forward and reverse estimators are conditionally unbiased and uniformly bounded" \
  --provided-assumption "the stale replay term is uniformly bounded" \
  --provided-assumption "the sum of step_size times replay_coefficient is finite" \
  --provided-assumption "Robbins-Monro step sizes are deterministic" \
  --provided-assumption "the hybrid mean gradient is strongly stable around phi_star"

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp check-proof-obligation \
  "a*Z*0+b*0" "0" --assumption "a > 0" \
  --assumption "b > 0" --assumption "Z > 0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp prove-or-counterexample \
  "a*Z*0+b*0 = 0" --assumption "a > 0" \
  --assumption "b > 0" --assumption "Z > 0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp check-proof-obligation \
  "z0*cos(pi/2)+p0*sin(pi/2)" "p0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp check-proof-obligation \
  "z0*0+p0*1" "p0" --backend sympy

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp audit-math-to-code \
  "normalized_weights=exp(log_softmax(log_weights)); negative_log_prob=-transport.log_prob(physical); loss=reduce_sum(normalized_weights*negative_log_prob)" \
  bayesfilter/inference/neutra_weighted_training.py

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp classify-math-claim \
  "The mathematical note proves that the existing finite SSL-LSTM q=20 dense IAF and 600 normalized SMC particles will train successfully and yield globally mixing HMC." \
  --evidence "The note labels finite model capacity, optimizer success, SMC-N bias, and HMC crossing as unproved empirical gates."

/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp prepare-review-packet \
  "Does the final adaptive-replay NeuTra note establish only conditional mathematical viability, including fixed-law replay, adaptive fresh blocks with summable stale replay, correct normalized-SMC boundaries, exact-minimizer logic, and explicit finite-SSL-LSTM nonclaims?" \
  --source '{"path":"docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md","role":"primary_mathematical_note"}' \
  --evidence '[{"kind":"backend_verified","summary":"SymPy certified only the nonzero deterministic-mixture denominator cancellation."},{"kind":"human_review_required","summary":"Measure theory, both stochastic-approximation theorems, Rosenblatt existence, and ideal-HMC arguments require independent review."},{"kind":"claim_boundary","summary":"Finite dense-IAF training, whitening, mode exploration, and HMC success remain unsupported nonclaims."}]' \
  --packet-id ssl-lstm-q20-adaptive-replay-neutra-math-r3-20260821 \
  --handoff
```

The final review packet was prepared with `prepare-review-packet`, packet ID
`ssl-lstm-q20-adaptive-replay-neutra-math-r3-20260821`, using the mathematical
note as the primary source and both backend-verified and
human-review-required evidence classes plus the finite-application claim
boundary. It returned `review_ready` while
explicitly denying that packet construction recertifies nested evidence.

## Manual audit and revisions

The tool results caused the following substantive revisions or retained
boundaries:

1. Theorem 1 is conditional on support, evaluable proposal densities, frozen
   proposal history, and integrability. Symbolic cancellation alone is not
   called an importance-sampling proof.
2. Normalized SMC blocks are classified as SMC-N and receive only a
   consistency claim. A future SMC-U route must prove its own unnormalized
   Feynman--Kac identity.
3. The unknown normalizer multiplies the forward term, so the hybrid objective
   is `a Z F + b R`, not silently `a F + b R`.
4. Theorem 2 now assumes a bounded Lipschitz Poisson solution, bounded update
   noise, an inactive projection along an interior trajectory, and explicit
   strong mean-gradient stability around a unique optimum. None of those
   application-specific properties is claimed for the current dense IAF.
5. Persistently replaying blocks produced by a proposal that adapts with the
   learned transport is excluded from Theorem 2. Theorem 2A separately proves
   continual adaptive fresh-block generation when stale replay has summable
   influence; constant-weight stale replay still needs a controlled-Markov,
   increasing-buffer, cross-fitting, or other finite-error proof.
6. Exact triangular transport existence is separated from membership in the
   implemented finite dense IAF.
7. Exact Gaussian pullback is separated from performance of the current
   finite-step leapfrog HMC kernel.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the refreshed-replay construction as mathematically viable under stated assumptions | Conditional identities and sufficient theorem are written without a known contradiction | No equality was refuted; full-proof certification was not obtained | Sufficiency and presentation of Theorem 2's Markov-noise argument | Independent theorem-level review by Fable | No finite q=20 training, whitening, mode exploration, convergence, or HMC-readiness claim |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `b130d09e19856ac75b7bf9187d809e5f8a7a23b2` with unrelated pre-existing worktree changes preserved |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; Python 3.13.13; MathDevMCP CLI |
| CPU/GPU | CPU-only symbolic/document audit; no TensorFlow import and no GPU initialization |
| Data | N/A; no posterior data or particle artifact was read by MathDevMCP |
| Random seeds | N/A; deterministic symbolic routes plus bounded deterministic claim routing |
| Wall time | Not used as scientific evidence; each command completed in seconds |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematics-review-plan-2026-08-21.md` |
| Result | This audit ledger and the linked mathematical note |

## Document verification

- Pandoc parsed all four deliverables from GitHub-flavored Markdown to plain
  text without error.
- All four deliverables are ASCII-only and have no trailing whitespace.
- Every fenced-code count is even.
- The two local paper texts and both implementation anchors referenced by the
  note exist at the recorded paths; the cited line regions were inspected.
- The original audit scope check showed only the four mathematical-audit
  documents created in that pass. The later Fable reply and this adjudication
  amendment are recorded separately. All unrelated modified files remain
  untouched.

## Post-audit red team

The strongest alternative explanation is that the proposal is mathematically
sound only in a deliberately strong fixed-law, strongly stable setting that is
too far from adaptive dense-IAF training to guide the actual repair. Evidence
that would overturn the positive conclusion is a valid counterexample to
Theorem 1, Lemma 1, Theorem 2, or Theorem 2A under all of the corresponding
assumptions. The weakest parts of the current evidence are the two stochastic-
approximation proofs, especially the Markov-noise argument; they are the
primary Fable review targets.

## Post-Fable disposition

Fable's independent review is preserved at
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-review-reply-2026-08-23.md`.
It returned `AGREE` for the conditional mathematics and identified textual
repairs plus a symmetry-based local-basin boundary. The review does not
upgrade this MathDevMCP ledger to a machine proof certificate and does not
establish runtime boundedness, dense-IAF capacity, optimizer success, or HMC
validity. The adjudication and plan amendment are recorded at
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.

After the revision, `mathdevmcp doctor` again returned `ok: true`, and the
deterministic-mixture cancellation again returned `proved` through SymPy. A
bounded `assumptions-for` query about global strong monotonicity in the
symmetry-rich dense IAF returned `inconclusive` because that route is outside
MathDevMCP's rule set; it neither refutes nor certifies the manually derived
symmetry obstruction in the note.
