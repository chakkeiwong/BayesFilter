# Phase 42 Repair and Refresh

Date: 2026-08-26  
Source result: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase42-result-2026-08-26.md`

The v2.4 hard boundary passed. The report nevertheless found a material
bank-to-bank support difference: bank B was descriptively improved while bank
A was an outlier. This is a promotion veto for whitening and a repair trigger,
not a continuation veto or a reason to select bank B.

The active plan is refreshed to `v2.5-third-bank-support-diagnostic`. The next
smallest discriminating artifact is one fresh N=256 bank (seed
`(20260826, 4104)`) evaluated together with unchanged banks A and B after one
trainer per arm. The new runner must verify that the reconstructed terminal
state hash equals the corresponding v2.4 state hash. Fresh rows remain false
for training and selection. No objective, architecture, proposal, or
whitening criterion changes.

The v2.4 result remains immutable evidence; no bank is pooled, removed, or
retroactively relabeled.
