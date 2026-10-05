# Adaptive iAPF decision localization

The new native decision primitive remains **unadmitted**. The existing adaptive
adapter and selection validator are unchanged. CPU12 decision cases and7
invalid-input cases pass04867 after a localized compilation repair, but GPU
04868 fails the strict stopping boundary.04870 passes161 checks that verify
the failure and diagnosis; that readback does not accept the primitive.

For the frozen boundary history, the original Python CV is
0.3303714383398673. At the next FP64 threshold0.3303714383398674, the original
stops but the initial compiled CPU candidate continues.04864/04866 establish
that shifted logs, exponentials and sequential sum match on CPU. Constant
division by3 changes the mean by one unit; optimized HLO explicitly substitutes
multiplication by0.33333333333333331. Runtime divisors restore the intermediates,
but HLO rewrites the final nested quotient as `(root * count) / total`.
Isolated scalar division itself agrees exactly with Python.

04865 shows local optimization barriers alone do not fix the constant reciprocal.
Using actual valid-window cardinality plus local division barriers preserves
the original denominators/formula and passes the unchanged CPU gates04867.
The cardinality equals k+1 on every valid complete window; absent history does
not contribute. No global compiler flag or statistical threshold changed.

GPU04868 still returns CV0.3303714383398674.04869 localizes the first discrepancy
to `exp(-0.19999999999999998)`: Python yields0.8187307530779818 and GPU yields
0.8187307530779819. Both the instrumented calculation and ordinary owner
reproduce the GPU CV. The GPU's isolated scalar divisions agree with Python.
This is a backend transcendental-rounding difference at a discontinuous decision,
not evidence of a broken variance formula or justification to waive a branch.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Retain CPU compilation repair |12 decisions/7 invalid cases pass | No CPU boundary flip remains in fixture | Broader k/history scope | Preserve and extend qualification | No full adaptive migration |
| Do not admit GPU primitive | Near-threshold action differs | Strict action mismatch | Numerically unresolved comparison | Evaluate explicit fail-closed resolution status | No threshold relaxation |

| Inference status | Finding |
|---|---|
| Hard veto | GPU strict boundary still fails |
| Statistical ranking | None |
| Descriptive findings | One-unit exponential difference and its propagated CV |
| Default readiness | Open; active adaptive runtime unchanged |
| Next evidence | Reviewed resolution guard, then full controller/ledger/endpoint repair |

All commands, arrays, HLO, sources and provenance are in runs04863–04870. Eight
workers consumed32.810664463 CPU and11.402536258 GPU seconds, within the final
600/600-second allocation. Remaining global budget is25.008236 CPU/24.772866 GPU
process-hours. The plan is `filter_gradient_iapf_controller_20260929.md`; original
source/fixture and rejected initial candidate are preserved under
`tests/fixtures/filter_repair_iapf_controller_20260929/`. New source/harness lint
and whitespace checks pass. Guard coverage is313 sources/1457 exceptions; the
new native component introduces no exception.

Post-run review: exact branch compatibility at adjacent floating values is
stronger than ordinary smooth-value parity. The tests exposed that difference
before changing the controller. Emulating host libm on GPU would add a new
transcendental authority and is not the recommended execution repair. Evaluate
a declared numerical-resolution veto under the repository's Class B fail-closed
guard policy, with no-fire checks on resolved histories and explicit rejection
of ambiguous ones. Such a veto must report an error rather than select a new
action; preserve this original incompatibility as evidence. Final shape dispatch,
fit/cast validity, timing/work accounting, ledger validation and actual public
consumers remain separate obligations.
