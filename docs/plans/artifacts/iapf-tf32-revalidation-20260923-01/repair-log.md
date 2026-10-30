# Local repairs

Attempt02 (R fixture preparation) rejected a matrix passed to the reference
FP32 serialization helper: base R `writeBin` requires a vector. No scientific
computation or GPU worker had run. Repair the diagnostic matrix wrapper by
vectorizing in R column order and restoring its original dimensions. The
reference helper and numerical consumer stay unchanged. Verify the repaired
wrapper preserves shape, orientation and expected FP32 representations, then
retry in a fresh source/fixture directory. The failed 0.1662 CPU seconds and
first 0.4712-second old-fixture attempt remain charged. Evidence contract,
seeds, tolerances and total budget are unchanged. Skeptical repair review: PASS.

The input-representation audit during attempt08 found that R's decimal CSV
writer does not always preserve the exact promoted FP32 bit pattern when
reloaded directly into FP64: 6,396 of 8,811 finite common inputs differ by at
most 5.1071e-15. This is negligible at the comparison tolerances but fails the
plan's exact common-input statement. Repair the distributional loader to
construct FP32 tensors first and cast them to the requested dtype. Leave the
old unrounded R parity fixture loader unchanged. For FP32 and TF32 this is
exactly the existing float-to-FP32 conversion followed by an identity cast;
their running/completed results remain applicable. Only the FP64
distributional run and the combined summary need replacement. Preserve the
old FP64 and summary as superseded harness evidence, charge their full time,
then rerun in fresh attempts11/12 with the same seeds, budget and criteria.
No shared algorithm source changes. A host CPU loader regression must verify
all common values recover their FP32 bit patterns and FP32 inputs are unchanged.
Skeptical repair review: PASS; this restores the original input contract.
