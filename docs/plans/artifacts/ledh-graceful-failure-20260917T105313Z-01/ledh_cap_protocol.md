# LG1 coordinate-cap non-harm calibration

Static inspection of `higher_moment_contract_e.py:1462` shows that the positive
standardized cap applies to particle coordinates, followed by affine moment
restoration. The inherited value 0.98 was only a warm hypothesis and is
inappropriate to promote without a target-specific check: a Gaussian has
substantial mass beyond one standard deviation. Restoring the first two moments
cannot undo tail-shape deformation. This is an identified default-assumption
flaw, not evidence that the entire LEDH method is invalid.

For standardized coordinate z, the implemented map is
z / (1 + (z/c)^8)^(1/8). Its relative displacement is bounded above by
(abs(z)/c)^8 / 8, by convexity of (1+u)^(-1/8). This supplies a control scale
independent of likelihood or posterior performance. Test c={0.98,2,4,8,16,32}
on a fixed N4096 randomized Gaussian cloud with exact empirical zero mean/unit
variance. Use the full shared correction with source=reset cloud, keeping all
other controls fixed. Zero cap is an explicitly diagnostic ablation comparator;
it cannot become the canonical runtime configuration.

Non-harm criterion: a valid healthy output with maximum displacement <=1e-4 in
standardized units relative to the diagnostic comparator, plus finite flagged
behavior when one reset input coordinate is 1e4. Emit cap displacement and
pre/post magnitudes. Choose the smallest positive passing cap, never the best
likelihood. Report post-restore skew/kurtosis to expose shape changes. The smooth
map is not exactly an identity in real arithmetic; this is a precision-bounded
non-harm screen, not a claim of exact equality or universal safety.

This is a dedicated numerical-control evaluation, no new substantive filter
attempt or holdout access. GPU/TF32/XLA, 180-second timeout within Phase 4.
If no value passes, block the final filter repair until this prerequisite is
resolved. A passing cap still needs scope-specific downstream calibration,
validation and derivative checks. It is not a new global default.
