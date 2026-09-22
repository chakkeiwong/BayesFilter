# Bounded review of the SGQF initialization conclusion

Question: do the statements under “Conclusion” follow from these recorded
facts, without a filter-quality or statistical-superiority claim? This summary
is self-contained. Review interpretation only, not uninspected implementation.

## Facts

The owner requested SGQF initialization and an SGQF accuracy safeguard. A04's
numerical protocol was independently reviewed AGREE before execution. The run
completed 24 cases: d1/d4, t1 with Gaussian incoming law and t1/t17/t18 with
saved recursive TT incoming law, three fresh train/validation/audit row designs.
The observations were already exposed; there are no independent observation
replications or uncertainty intervals. Each within-case initialization arm
uses the same target, row design, degree/rank, scalar normalization, L1 grid
and solver budget. Target-source hashes match the old saved campaign. Eighteen
CPU-only tests and all per-case GPU/XLA validity checks passed.

Median audit Hellinger discrepancies for d4:

| Target | Exact SGQF | Generic-start fit | SGQF-start fit |
| --- | ---: | ---: | ---: |
| t1 Gaussian incoming | .001866 | .009101 | .007671 |
| t1 saved TT incoming | .005701 | .010610 | .008716 |
| t17 saved TT incoming | .010001 | .005630 | .005092 |
| t18 saved TT incoming | .204194 | .121157 | .090554 |

d1 initialization differences are below 1.51e-9. Against the d4 t1 Gaussian
target, converting SGQF to degree-3/rank-3 TT gives median error .003206 before
the 5% defensive component and .005384 after it. The component is inherited,
uncalibrated, and has not been removed or promoted based on this comparison.

A validation selector includes exact SGQF, the conversion and both fitted
families. Its chosen validation error is no larger than SGQF in 24/24 cases,
by construction. Choice is saved before independent audit rows are generated.
Audit is no worse in 23/24 cases; the exception is .000017008 worse. At d4 t18
the selector chooses generic in 3/3 cases despite SGQF-start being lower on
audit. Audit is not reused for selection. No filter-level fallback was installed.

## Conclusion

The d4 median differences nominate SGQF initialization as a candidate, with
all rankings remaining descriptive. At **t1 only**, exact SGQF has lower
observed audit error than either fitted TT on both incoming-law targets. At
t17 and t18, the fitted TTs have lower observed error than SGQF. This is a
mixed outcome; no all-times SGQF dominance is claimed.

A04's predeclared rule is that heuristic underperformance in ANY evaluated
required case is a promotion veto. Passing other cases does not remove it.
This follows the owner's policy that losing to any cheap heuristic in any
salient situation blocks promotion. Thus the t1 failures veto promotion of
this TT configuration while t17/t18 keep the correction scientifically viable.
The veto is not a continuation veto for research.
The proposed safeguard can preserve an empirical validation baseline but
cannot guarantee unseen accuracy. The exact SGQF candidate should be retained
in the next protocol; later targets show why forcing SGQF everywhere is also
unjustified. Saved TT incoming laws may inherit earlier error, so fitting them
does not certify agreement with the true filter. Dedicated defensive-mass
coverage/non-harm calibration and independent-sequence filtering remain open.
The coefficient initializer is diagnostic and exponential in dimension; no
scalable/default runtime, global optimization or filter-quality claim follows.

Return material interpretation blockers only, at most 250 words, and end with
VERDICT: AGREE or VERDICT: REVISE.

## Disposition of the first bounded review

The first response returned REVISE, applying the t17/t18 rows against a claim
explicitly about the first transition (t1). It also said two favorable cases
remove a promotion veto from two unfavorable cases; that conflicts with A04's
case-conditional rule, clarified above. No numerical result or decision rule
has changed. Please check the t1-only statement and the stated veto rule, not
an all-times dominance claim that the result does not make.
