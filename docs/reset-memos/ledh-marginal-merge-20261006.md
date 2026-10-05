# Active marginal-weight integration checkpoint

Question: minimal local shared correction, reproducing old LGSSM/KSC and the
other branch's range-bearing calculation. Stage: implemented and checked on
sqmc-development at base 0b91a64f6 (uncommitted focused changes).

The owner's direct-patch instruction supersedes the earlier broad integration
worktree/dependency-extraction plan. Active plan:
`docs/plans/ledh-marginal-minimal-patch-2026-10-06.md`.
Results: `docs/benchmarks/ledh-marginal-minimal-patch-results-2026-10-06.md`.
Artifacts: `docs/plans/artifacts/ledh-marginal-minimal-patch-20261006-01`.

Checked: one Gaussian density/tangent implementation with diagonal/all-component
selection; flow matrix and derivative supplied by the existing flow loop; both
batch wrappers forward the policy. Default stays ancestor. N=64, T=1/20,
two points/model, FP64 CPU/XLA: LG3/KSC old-value and score agreement <=3e-14;
M13 marginal vs pinned f5e69d716 <=1.7e-13 value and <=2.3e-12 score. All 24
patched evaluations finite. 35 distinct tests pass after test-fixture repairs.
Exact multi-tile check N=4096,K=2048 and batch call-chain checks pass.

No scientific promotion, GPU/FP32 conclusion, main merge or push. Existing dirty
Zhao-Cui research documents and untracked campaign artifacts preserved. Current
source branch live edits were avoided via pinned snapshot in /tmp.

Budget: at most 900 of 1800 planned CPU engineering job-seconds charged
conservatively; serious 48-hour extension budget unspent by these mechanics
checks. Original reference campaign remains 169,213.099 job-seconds.

Next action: present the completed small patch and numerical agreement. Resume
PP/SIR canaries from this implementation under their existing separate evidence
contract; inspect checkout/checkpoint before running. Main synchronization remains
a later concrete action, not authority to merge the unrelated source branch.
Independent Zhao-Cui continuation: zhao-cui-publication-replication-20261004.md.
