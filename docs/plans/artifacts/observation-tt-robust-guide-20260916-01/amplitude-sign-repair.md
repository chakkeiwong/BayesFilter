# A10 localized amplitude-sign repair

## Failure and mathematical diagnosis

Confirmation-01 d4-s05, stable arm, t19 fails at warm.fit_step's assertion that
the least-squares scalar is positive. Its value is -0.3106276392630666. The
full mixture then has no fitted stable TT. Baseline and repaired-guide arms
complete. Preserve that failed confirmation; this is a fitting implementation
restriction, not evidence against the physical mixture identity.

For initial amplitude h, nonnegative row weights w and target amplitudes b,
the scalar least-squares minimizer is c=sum(w*h*b)/sum(w*h*h). It is allowed
to be negative even when b is positive because a finite TT projection may
have either sign on the rows. For nonzero c the normalized squared density
is unchanged by replacing h with c*h: c^2 cancels in its normalizer. In
particular, the amplitude sign is unidentifiable from the represented density.
The valid requirements are a finite positive denominator and finite nonzero
c, not c>0. A zero scalar still cannot initialize a nonzero TT and must be
rejected explicitly. No clipping, ridge, epsilon change or target change.

## Evidence contract and bounded retry

Question: does removing the unjustified sign restriction permit the same
least-squares algorithm to execute, without changing healthy positive-scale
outputs? Baseline is the current positive-scale implementation. Primary repair
acceptance is exact positive-branch parity, sign-flipped-initializer equivalence,
unchanged normalized squared density, and completion of the exposed failed fit.
Zero/nonfinite denominator or scalar remains a validity veto. Fresh filtering
quality continues to use A10's original criteria; repair success alone cannot
promote stable coordinates or the mixture.

After confirmation-01 finishes and its sources/results are archived:

1. Add a tested initializer scaling helper used by the actual fit endpoint.
   Keep the arithmetic and returned values identical for positive scalars.
2. Document the signed least-squares proposition/proof and request a bounded
   MathDevMCP audit of it. Retain tool limitations honestly.
3. Smoke-03 adds the exposed d4-s05 data AND its original fitting seed to the
   existing diagnostics. It is exposed debugging data, not fresh validation.
4. Calibration-02 repeats the existing calibration data and selection rules.
   Compare its frozen controls and positive-scale evidence with calibration-01.
5. Confirmation-02 uses fresh blocks 48--71, disjoint from earlier calibration
   0--5 and confirmation 24--47. No tuning on confirmation-01 results. The
   same 12 sequences per dimension, references, intervals and vetoes apply.

This uses the remaining one smoke, one calibration and one confirmation slot
under A10. Each remains capped at 600/5400/5400 seconds; the total numerical
18000-second and active 28800-second ceilings remain binding. No new compute,
method family, criterion, hardware class or authorization is required.

## Skeptical review before editing

PASS as a localized implementation repair. The assertion excludes a valid
least-squares solution solely by its amplitude sign; the proof provides a
principled correction. It must not be described as improved approximation or
default readiness. Positive-case parity guards comparability; fresh seed
blocks prevent the exposed failed fit becoming confirmation evidence. Preserve
all failures and original inference. Broader-chart capacity, reference error,
rare failures, heuristic losses and full-program derivatives remain unresolved.
This is executor self-review, not an independent reviewer verdict.

Implementation checks: tests-05.log records 38 passing tests in 8.21 seconds,
including positive-branch bitwise parity, negative/global-sign equivalence,
unchanged normalized squared density and explicit zero-scale/norm rejection.
tests-04.log is a preserved command-path error (no tests ran), repaired by
using the actual test file names. Compile checks pass. LaTeX build-pass4 passes.
MathDevMCP mathdev-amplitude inspected the new equation and returned a
formalization-role abstention, no counterexample or concrete correction. Its
scope is partial and cannot certify the proof; the direct least-squares
derivation and density identity above remain the checked mathematical basis.

Confirmation-01 completed in 1211.579978 seconds with all 24 references valid;
stable/full each lack exactly the d4-s05 fit. Original result, inference and
exact dependency sources are preserved in comparison.json and source-snapshot/.

Smoke-03 completed in 236.078364 seconds. The exposed d4-s05 t19 fit retains
the original scalar -0.3106276392630666 and now completes. All three protected
arms on both healthy cases and all three exposed cases have zero inverse-CDF
bracket failures and invalid consumer steps. The original guide failure on
A09 d4-s10 remains visible. This passes the localized repair criterion; it
does not establish filtering accuracy. Proceed to calibration-02 and fresh
confirmation-02 under the unchanged protocol.

Calibration-02 completed in 551.950738 seconds with all six reference checks
passing and all 90 arms complete. Its entire selected-controls.json is equal
as parsed JSON to calibration-01, including the full selection ledger and
reported MSEs. The correction therefore changes no calibration selection or
positive-scale result in these cases. Confirmation-02 uses these frozen
controls and fresh blocks 48--71. It completes all TT arms on all 24 cases
without numerical/CDF/log-evidence failures. One imprecise d4 reference and
the declared conditional heuristic losses prevent filtering promotion; these
do not invalidate the localized signed-scale correction. Final confirmation
has no negative initialization scales, so smoke-03 remains the direct replay
evidence for that branch. See the A10 result and result-review.md.
