# M22 sequential null confirmation

Both predeclared null arms pass the operating-characteristic screen. All 512
whole experiments per arm were valid, and each pointwise exact 95% rejection
interval has upper endpoint below 0.10. This closes the previously inadequate
interval precision for this frozen Gaussian experiment. It does not establish
exact 0.05 size, general target coverage, posterior stopping coverage or power
against subtle defects in complete tuning and sampling fits.

| Arm | Valid / planned | Rejections | Fraction | Exact pointwise 95% interval |
| --- | ---: | ---: | ---: | --- |
| Baseline | 512 / 512 | 25 | 0.048828 | [0.031845, 0.071239] |
| No-op | 512 / 512 | 19 | 0.037109 | [0.022487, 0.057346] |

The design is the fixed inventory in
`bayesfilter-hmc-merged-source-continuation-2026-09-22.md`, preserving the M22
screen and M16 Gaussian experiment: epsilon 0.3, L=5, 32 complete transitions
per powered step, seven rank draws, initial 16384 independent anchors, up to
three independent looks and a sample multiplier of two. Root seed is
2026092243. Pilot outcomes are excluded. No extra trials were added after
seeing the result.

The terminal audit checked all 1024 experiment records and all 1298 looks,
including finite substep states/log ratios, distinct look seeds, complete
sample counts, unchanged test family, and reconstruction of each sequential
decision from raw p-values. It independently recomputed the Clopper--Pearson
intervals by inverting binomial sums. Look counts were 787 experiments with
one look, 200 with two, and 37 with three. No invalid outcome was silently
treated as a nonrejection.

The frozen source matches the Git archive of `f9c86f41a` byte for byte for every
Python file. Full-package identity is
`13917d3625fa5a09dfd3bf1ab9d2894618ed7bc4259ca7603f16c766f6213d66`;
the runtime Python-only identity is
`b8507f7ae75d10e63db6d8245105d0467bf2382a5d667857507241758107c303`.
The full manifest additionally includes `symmetric_sylvester_op.cc`, explaining
the different hashes; there is no Python-source discrepancy. These results do
not certify the later bootstrap/public-diagnostic repair or supplied-map edits.

The numerical worker used the tfgpu environment (Python 3.13.13, TensorFlow
2.20.0, TFP 0.25.0), GPUs deliberately hidden and one intra/inter-op thread.
Worker charge is 5879.119874261902 CPU seconds; the read-only terminal audit
cost 80.8613317910349 CPU seconds. No GPU time was used. Commands, environment,
attempt logs, designs, runtime provenance, exact arrays/statistics and result
checksums are in `artifacts/hmc-repair-master-2026-09-16/m22-r2/`.
`null-confirmation-audit.json` is the compact checked summary;
`audit_null_confirmation.py` records the independent audit. Coordinator time
is not charged again.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this null-size precision cell | Both fixed-denominator upper endpoints <=0.10 | All experiments and look/source inventories valid | Only this Gaussian frozen-kernel law and design | Preserve result and finish independent M21 work | Exact nominal size or universal calibration |
| Keep subtle full-fit power open | No adequate repeated full-fit power study | No invalidity in the completed null study | About 20 CPU hours per 128-dataset arm at measured pilot cost | Resolve affordability/efficiency before a separately fixed full-fit inventory | Null nonrejection proves defect sensitivity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No invalid experiment, reused look seed or source discrepancy |
| Statistically supported ranking | None; a baseline/no-op difference was not a comparison objective |
| Descriptive-only differences | Rejection fractions and look counts between arms |
| Default readiness | No tuning, posterior or validation default changed |
| Next evidence needed | Stopped posterior coverage and subtle complete-fit defect power remain separate |

Post-run review: the strongest alternative explanation for apparently good
behavior is that this tractable Gaussian law does not stress the difficult
targets or full preparation/tuning procedure. That limits transfer, not the
observed null-rate result. A new target-specific size or validity failure
would overturn a broader claim; no such broad claim is made here. M21 continues
with the same frozen source and its predeclared denominators.
