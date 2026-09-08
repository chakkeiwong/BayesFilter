# MathDevMCP audit: LEDH-PFPF-GenUT dual-cap note

Date: 2026-08-24  
Status: `COMPLETED_WITH_LIMITS_NO_CONTRADICTION_FOUND`

## Audit scope

The audit covers the proposition-proof note
`docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md`, its narrow TeX audit appendix, and the four local implementation paths named in the execution plan. The question was whether the note contains a mathematical contradiction and whether its conditional solution is being presented as a proof about the current implementation.

This is a diagnostic audit, not a certification of the LEDH implementation, replay estimator, NeuTra training, mode coverage, or HMC readiness.

## Run manifest

| Field | Value |
|---|---|
| Repository commit at audit start | `14e4618749c9e04e8c4d2398becadb0206b30599` |
| Python | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, 3.13.13 |
| MathDevMCP | `/home/ubuntu/python/MathDevMCP/src`, CLI module |
| SymPy/Sage | available according to `doctor` |
| LaTeXML/Pandoc | available according to `doctor` |
| Lean | unavailable; no Lean proof was requested or used |
| GPU | not used; this was a documentation and symbolic-check run |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-literature-solution-plan-2026-08-24.md` |
| Markdown SHA-256 | `83a973fbb7cd23c8f8427e6a325847df67fe4de46fcc82b8caf8c90f49089d85` |
| TeX SHA-256 | `885c7f79af5e5849431da393a0f999fce8fcc51a9048d70956824249db038e41` |
| Output root | `docs/plans/artifacts/ledh-pfpf-genut-literature-solution-20260824/` |

## Commands and outcomes

### 1. Environment check

Command:

```text
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src python -m mathdevmcp.cli doctor
```

Outcome: `ok: true`. SymPy, Sage, LaTeXML, and Pandoc were available. Lean was
unavailable and was not required.

### 2. Markdown document screen (final source)

Command:

```text
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src python -m mathdevmcp.cli audit-applied-math-document \
  docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md \
  --mode screen --specialist-policy none --response-mode detailed \
  --artifact-root docs/plans/artifacts/ledh-pfpf-genut-literature-solution-20260824/mathdevmcp/document-screen-final2
```

Outcome: exit 0, `completed_with_limits`, `finding_count: 0`. Artifact:

`mathdevmcp/document-screen-final2/audit-a80fd2f9e6a8c5ae311f805beb9c9fcaa91fb3315d3387eef0218f06209560de.json`

The tool warned that no code paths were supplied, so implementation alignment
was not checked in this pass.

### 3. Markdown deep pass with code paths (final source)

The same command was run in `--mode deep` with these exact code paths:

```text
bayesfilter/highdim/genut_shape_lm_tf.py
bayesfilter/highdim/dual_cap_genut_primal_tf.py
bayesfilter/highdim/genut_guided_proposal_tf.py
bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py
```

Outcome: exit 0, `completed_with_limits`, `finding_count: 0`. Artifact:

`mathdevmcp/document-code-deep-final2/audit-fdd6ba0a12caa48b4b53cbdf7cf60f91fceb421cb5271a0ba5e6635a50d5e6fd.json`

The 12 displayed obligations were all `not_checkable`: the document equations
are Markdown/LaTeX text and were not converted into the backend's typed
obligation form. This is an abstention, not a proof of the claims.

### 4. TeX appendix deep pass

The machine-auditable appendix was compiled with `latexmk` and then audited:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=docs/plans/artifacts/ledh-pfpf-genut-literature-solution-20260824/latex \
  docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.tex

PYTHONPATH=/home/ubuntu/python/MathDevMCP/src python -m mathdevmcp.cli audit-applied-math-document \
  docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.tex \
  --mode deep --specialist-policy none --response-mode compact \
  --artifact-root docs/plans/artifacts/ledh-pfpf-genut-literature-solution-20260824/mathdevmcp/tex-deep-final2
```

Compilation succeeded and produced a three-page PDF. MathDevMCP returned exit 0,
`completed_with_limits`, `finding_count: 0`; all six labels were found, while
eight selected obligations remained `not_checkable`. Artifact:

`mathdevmcp/tex-deep-final2/audit-64cdfa0615c82b0c4673947fc24fa02dbf4ade65aab04f57b4ab2b6352ba46cd.json`

The only LaTeX warning was an overfull box caused by the long audit-file name;
it does not alter the equations.

### 5. Isolated proof-label pass

To avoid unrelated malformed TeX elsewhere under `docs/plans`, the appendix was
copied to the isolated audit root
`mathdevmcp/../isolated-root/note.tex`. The six labels were located, but the
proof-audit-v2 adapter did not certify them:

| Label | MathDevMCP result | Meaning |
|---|---|---|
| `prop:ledh-moment-nonidentification` | `inconclusive:source_label_missing` | Parser found the label but emitted no typed obligation |
| `prop:ledh-cap-support` | `unverified:manual_formalization_required` | Norm/cap notation is outside the bounded algebraic backend |
| `prop:ledh-pfpf-change` | `unverified:manual_formalization_required` | Change-of-variables notation needs manual formalization |
| `cor:ledh-reset-boundary` | `inconclusive:source_label_missing` | Narrative corollary was not a typed obligation |
| `prop:ledh-replay-unbiased` | `unverified` plus `inconclusive` rows | Sums/measure notation was not safely extracted |
| `prop:ledh-defensive-bound` | `unverified:manual_formalization_required` | Mixture-density notation needs manual formalization |

An earlier repo-wide label invocation also hit MathDevMCP's `invalid brace
depth in LaTeX display environment` while indexing unrelated files. The isolated
rerun removed that environmental/parser collision but did not turn the semantic
rows into certificates. Neither outcome is evidence against the propositions.

### 6. Narrow symbolic certificates

The following `check-proof-obligation --backend sympy` checks returned
`status: equivalent` with a zero simplified difference:

1. `(1+r**2/(d*rho**2)) - r**2/(d*rho**2) = 1`, under `d>0, rho>0`.
2. `1+s-s = 1`.
3. `(1/6)*sqrt(3) + (1/6)*(-sqrt(3)) = 0`.
4. `(1/6)*3 + (1/6)*3 = 1`.
5. `(1/6)*9 + (1/6)*9 = 3`.
6. `1-eps+eps = 1`, under `eps>=epsmin`, `epsmin>0`, and `eps<=1`.

A direct derivative workflow using SymPy `Derivative(...)` returned
`inconclusive/human_review_required` because the CLI parser treated the
symbolic derivative as a callable symbol. The derivative is therefore not
reported as machine-certified; it is derived explicitly in the note and can be
checked by ordinary calculus or a future typed obligation.

## Findings and interpretation

| Finding class | Result | Consequence |
|---|---|---|
| Hard contradiction in the displayed elementary identities | None found | The note may proceed as a conditional mathematical analysis |
| Source/implementation alignment | Not certified | The code anchors and Li--Coates interpretation still require human/Fable review |
| Full-support claim for the caps | Explicitly refuted in the note | Do not use the cap determinant as a global flow density |
| Replay unbiasedness | Certified only under the frozen-mixture and positive-denominator hypotheses in the note | Existing normalized replay remains outside that theorem |
| Adaptive AMIS convergence | Not checked by MathDevMCP | Preserve the literature assumptions and do not promote finite-run results |
| NeuTra/HMC readiness | Not assessed | Requires separate target-specific experiments and canonical sequential-HMC gates |

The machine result is therefore **no contradiction found, with material
abstentions**. It is not `AGREE` for the current implementation.

## Static verification

- `pandoc --from=gfm --to=plain` converted the Markdown note successfully.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error
  -outdir=docs/plans/artifacts/ledh-pfpf-genut-literature-solution-20260824/latex`
  reports the appendix PDF up to date after the successful compilation.
- All locally cited implementation, monograph, and paper-copy paths exist.
- The new note, audit record, and Fable handoff are ASCII-only.
- No experiment, GPU process, package mutation, default change, or destructive
  operation was performed.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Nonclaim |
|---|---|---|---|---|---|
| Retain dual-cap as proposal candidate | Conditional note separates finite-cloud moments from density | No cap-as-flow promotion veto is active | Exact proposal density after any reset | Human review of code/source anchors, then affine density fixture | No density faithfulness |
| Design replay repair | Frozen deterministic-mixture identity is proved | Normalized-only replay is vetoed for unbiasedness | Adaptive schedule and finite-buffer policy | Specify AMIS metadata or SMC-U block | No finite-sample unbiasedness for current blocks |
| Design mode repair | Tempered invariant-kernel recursion is stated | No finite mode guarantee claimed | Mixing/cross-mode probability | Build a separate two-mode validation plan | No claim that modes are currently explored |
| Promote to NeuTra/HMC | Not met | Promotion blocked | Diffeomorphism, score scale, downstream posterior checks | New scoped training and sequential-HMC evidence | No readiness or superiority claim |

## Inference-status table

| Row | Status |
|---|---|
| Hard veto screen | No MathDevMCP algebraic mismatch; implementation and support assumptions remain unevaluated |
| Statistically supported ranking | None; no stochastic comparison was run |
| Descriptive-only differences | None used in this documentary audit; local cap/whitening diagnostics remain explanatory |
| Default-readiness | Not ready; the note explicitly keeps the route opt-in/candidate |
| Next evidence needed | Bounded Fable review, typed density fixtures, replay-mixture checks, two-mode bridge test, then target-specific training/HMC plan |

## Bottom line

MathDevMCP supports the narrow algebraic checks and found no contradiction, but
its parser abstained on the substantive measure-theoretic rows. The correct
scientific status is `conditional route possible; current route not
admissible`, exactly as stated in the note. No result here licenses a claim of
IID Gaussian whitening, exact filtering, global mode coverage, or HMC validity.
