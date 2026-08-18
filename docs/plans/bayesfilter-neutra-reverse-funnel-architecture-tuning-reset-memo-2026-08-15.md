# Reset memo: NeuTra reverse-funnel architecture tuning (2026-08-15)

The reviewed architecture/tuning campaign and its staged repair are complete.

Key state:

- Four cold-start architectures were independently LR/schedule tuned.
- All four contain the exact funnel map by construction.
- All eight cold-start confirmations failed because the root distribution was
  compressed; width 200 and root-preserving permutations did not repair it.
- A root-scale-only warm-up passed the exact proposal law.
- Joint one-stage training initialized from that same selected warm state passed
  under two fresh joint-training and audit seeds. Full end-to-end warm-up seed
  replication remains untested.
- The supported diagnosis is joint trainability/co-adaptation, not insufficient
  raw capacity and not full reversal alone.
- No HMC was run. The exact funnel proposal-law pass does not establish
  SSL-LSTM, posterior, or HMC readiness.

Primary files:

- plan: `docs/plans/bayesfilter-neutra-reverse-funnel-architecture-tuning-plan-2026-08-15.md`
- result: `docs/plans/bayesfilter-neutra-reverse-funnel-architecture-tuning-result-2026-08-15.md`
- artifacts: `docs/plans/artifacts/neutra-reverse-funnel-architecture-tuning-2026-08-15/`
- cell runner: `docs/benchmarks/run_neutra_reverse_funnel_capacity_2026_08_14.py`
- campaign runner: `docs/benchmarks/run_neutra_reverse_funnel_architecture_campaign_2026_08_15.py`

Next scientific step: replace the exact-funnel privileged warm-up with a method
that uses quantities available for an unknown posterior, validate that method
on additional known-law controls, and only then return to SSL-LSTM NeuTra.
