# A04 independent protocol review

Claude reviewed only the self-contained A04 protocol through the bounded
read-only wrapper, with escalated access. Health and file-read probes passed;
the substantive review returned **VERDICT: AGREE**, with no material blockers.
Full output:
`../plans/artifacts/observation-tt-sgqf-initialization-20260915-01/review-01.txt`.
This is a protocol review, not an implementation or result verification.

Codex disposition: accept. Specify the projection residual roundoff allowance
as 256 times float64 epsilon times coefficient count; never clip a larger
violation. Recurrence runs by total multi-index degree up to 2*d*degree
(24 in d4), not merely up to total degree three. Degree three is per coordinate;
the review's informal parenthesis about stopping at cubic terms was inaccurate
and does not change the correct recurrence in the plan.

User authorization is the request to try SGQF initialization plus H6's five-hour
cap. No new direction or compute boundary is crossed. Execution can proceed
under A04, subject to the predeclared executable validity checks.
