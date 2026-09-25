# Filter and gradient repair resume checkpoint

Current checkpoint through 04108, September 26. Prior pushed commit:
`5bbce48b5`. The SIR terminal follow-up passes 04061/04062 (34 checks per
backend) and refreshed costs 04063--04068. The mixed KR/TTSIRT public logarithms
now execute through default-XLA owners, with one shared log-density/potential
owner. Final dedicated tests 04092/04093 and consumers 04106/04107 pass.
Policy 04108 passes 129 checks across 270 guarded sources / 1,424 exact
allowances. See `filter_gradient_latent_sir_result_20260926.md` and
`filter_gradient_mixed_kr_transport_result_20260926.md` for exact evidence.

Fresh cost cohort 04094--04105 preserves values (maximum error 2.22e-16),
identical inputs and 3,391 source hashes. Sharing the log owner removes the
earlier observed 22 MiB CPU / 31 MiB GPU overhead: final XLA RSS is within
0.5/1.3 MiB of the preceding partially compiled wrappers. These are descriptive
fixture costs, not target-capacity or performance-ranking evidence. Historical
KR and obsolete private helpers retain diagnostic loops; active method coverage
does not imply the entire file is loop-free or canonical scientific admission.

Charges through 04108: 85502.823304 CPU / 77468.094682 GPU seconds, leaving
32.249216 CPU / 30.481085 GPU process-hours under unchanged 56/52-hour caps.
The extra 24 CPU hours are already included. No campaign worker is active.
Remote main `5e16df06f586c16bc58fb76bc62d4f6451e7690d` is contained in this
branch; main remains unmerged. Canonical LEDH rebuilding is excluded; the
current NeuTra architecture remains `bayesfilter_neutra_iaf_author_v1`.

Next: diagnose the GenUT one-of-216 FP32 cap-active report mismatch with
captured operands and high-precision arithmetic, without changing the 1e-7
predicate or comparison tolerance. Then qualify N*d*d capacity and continue
public-consumer/LEDH-reset and initializer/supervisor integration. Original
precision dispositions, repeated-constructor XLA retention, DZ5 graph replay
and GPU graph finite differences, external callback pfor, reporting/isotropic
proposals and final F01--F20 dispositions remain open. Main merge remains
blocked until the master gate passes. Pending reporting proposals are not
approved by elapsed time.
