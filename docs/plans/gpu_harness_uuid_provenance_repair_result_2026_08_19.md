# GPU Harness UUID and Provenance Repair Result

Date: 2026-08-19
Cross-repository result: `/home/chakwong/MacroFinance/docs/plans/gpu_harness_uuid_provenance_repair_result_2026_08_19.md`

BayesFilter's scoped repair is implemented. Focused policy/provenance/probe
tests passed (`71 passed`), shell and Python syntax checks passed without
bytecode writes, and the trusted UUID-pinned TensorFlow probe passed on the
RTX 4080 with shared memory-growth policy, matching NVIDIA UUID/name/PCI
identity, `/GPU:0` placement, and positive allocator bytes.

The RTX 5080 native probe and live selector retry were blocked by the external
approval reviewer timing out before process launch. This is unverified, not a
GPU or code failure. MacroFinance's equivalent UUID-pinned probe passed on the
RTX 5080, so no cross-repository CUDA visibility failure was inferred.

Changed scope is limited to `bayesfilter.runtime` GPU provenance/memory policy,
the named current selector-backed GPU wrappers, the native TensorFlow probe,
and focused tests. No filtering, likelihood, HMC, tuning, sampler, or
scientific run was launched, and existing dirty user changes were preserved.
