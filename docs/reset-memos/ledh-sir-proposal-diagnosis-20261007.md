# Active covariance-proposal checkpoint

Current question: how are beta_0, beta_G and beta_L selected, and are transforms
or proposal densities mixed? Completed 2026-10-08 on sqmc-development, base
209223fdd061fa39726c69550f39e2ee324f489c.

Checked: the old text omitted selection. The shared chapter now explains
categorical branch sampling and supplies Algorithm 0: independent-pilot convex
second-moment fitting, a declared transition floor, projected-gradient solving
with a gap certificate, independent validation and frozen probabilities.
The objective holds incoming clouds/maps fixed; it does not optimize the
analytical score or the whole recursive filter.

Both documents compile and use the same body. Section 27.5 in docs/main.pdf
(647 pages; printed 252–255 / PDF 272–275) and Section 5 of the standalone
(23 pages; pages 8–11) contain the addition. All 49 existing labelled equations
and three algorithms are unchanged; eight equations and Algorithm 0 are added.
All labels/citations resolve. Rendered new pages passed inspection.

MathDevMCP extracted no usable targets in the focused audit. It proved one
scalar cancellation; two derivative requests were not encodable. Direct
SymPy checks confirmed those derivative identities. This is partial algebra
coverage, not a fully formal proof or performance validation.

Evidence:
- docs/plans/ledh-proposal-beta-calibration-results-20261008.md
- docs/plans/ledh-proposal-beta-calibration-20261008.md
- docs/plans/artifacts/ledh-proposal-beta-calibration-20261008-01/
- docs/chapters/ledh_covariance_proposal_body.tex
- docs/benchmarks/ledh-sir-proposal-diagnosis-20261007.md (earlier diagnosis)

The prior checkpoint is preserved in this artifact directory's baseline.
Earlier integration and MathDevMCP reports remain historical stage records.
Unrelated dirty diagnostics and document work remain preserved.

Remaining limits: the composite proposal and new beta procedure are not runtime
implementations; no numerical beta has been fitted or validated. No new model
test, GPU job, scientific promotion, commit or push occurred. The SIR diagnostic
campaign remains closed. Documentation work finished within its 45-minute
budget using five TeX passes per document. No experiments are queued.
Human readability feedback is pending.

Exact next action: explain the probabilities and the new selection rule to the
user, linking the compiled section. Further implementation would need its own
bounded plan and full recursive value/all-score validation across protected
model/horizon scopes.
