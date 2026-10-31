# Walnutpie Git integration — 30 October 2026

Requested work: commit the completed standalone Walnutpie work, merge remote
main, resolve conflicts, and push. This is repository integration, not a new
sampler experiment or scientific promotion.

## Scope and pre-execution audit

The scoped commit is `8821e8c0c431071578084164c6915cb95fbb3d82`, based on
`0a1bcd26b6d752c13849435ed78d664745687a8f`. It adds initialization/run control,
R2P, supplied diagonal/dense mass, author replication, the hybrid prototype,
tests, reviewed plans and compact evidence. Multi-gigabyte draw arrays, HLO
dumps and duplicate source snapshots remain local. Existing prebuilt C++
objects remain in history.

The shared branch has no upstream; origin's default branch and the repository's
integration convention target `main`. The shared worktree has 106 unrelated
dirty tracked files. They were fingerprinted and left unchanged. No stash,
reset, cleanup or force-push is used. Integration takes place on
`integrate/walnutpie-20261030`, in `/tmp/BayesFilter-walnutpie-integration-20261030`.
The fetched remote parent is `a005760ed7e2c80c421bf42568f601fe1c850515`.

Skeptical audit: merging in the shared checkout could overwrite unrelated work;
the isolated worktree avoids that risk. A successful textual merge cannot prove
numerical compatibility, so focused regression tests precede push. Unresolved
conflicts, unexplained integration failures and unsafe history rewriting veto
push. No experiment, environment install or numerical default change is needed.

## Resolutions

- `AGENTS.md`: retain the local standalone benchmark exception, the owner's
  broader NUTS exclusion, and remote NAF/IAF architecture rules. Clarify the
  October 6 NAF study scope using the current owner instruction.
- Modern-HMC chapter: retain the local rejected-proposal bookkeeping explanation
  and the remote implementation/results section. No equations or results removed.
- Modern-HMC master: retain the newer October 27–28 continuation and accounting;
  preserve the October 20 closeout as historical.
- Modern-HMC checkpoint: retain the newer local checkpoint and append the remote
  October 20 checkpoint under an explicit historical heading.

The merge changes no Walnutpie package source from the scoped commit.

## Integration compatibility repair

Two SIR bridge checks passed on the exact pre-merge source but failed after
merging: the general simulator's compiled recurrence changed floating-point
rounding, violating the existing frozen dataset hash. The remote already
contains `bayesfilter.highdim.sealed_sir_dataset_tf.sealed_sir_tensors`. The
frozen SIR target now reads that authority instead of regenerating sealed data.
All original state, observation and time-order hashes remain unchanged and
checked. This preserves the original target; it does not relax data identity.
The same localized repair existed as unrelated uncommitted work in the shared
checkout; its file was neither staged there nor changed.

## Validation

Interpreter: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` (TensorFlow 2.20,
Python 3.13). GPU intentionally hidden for these CPU reference/regression tests.
All pytest commands use:

```sh
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q <files>
```

| Check | Result | Elapsed |
|---|---|---:|
| Walnutpie GPU-TF, gap closure, supplied mass, hybrid, author funnel and million replication test modules | 76 passed | 262.33 s |
| Modern-HMC rejected proposals, kernels, bridge, MacroFinance resume, saved precision and saved tail modules | 105 passed; two SIR hash failures identified | 89.47 s |
| The two SIR bridge cases on archived pre-merge source | 2 passed | 9.56 s |
| Full bridge and SIR target-design modules after sealed-data repair | 38 passed | 51.07 s |
| Staged Python/JSON syntax | 33 Python and 39 JSON files parsed | routine check |
| Merged modern-HMC survey | 47-page PDF built; changed pages 38–39 inspected | routine check |

The HMC/SIR checks additionally use
`BAYESFILTER_OP_LIB_DIR=/home/ubuntu/python/BayesFilter/bayesfilter/ops`. The
first isolated attempt stopped at collection because the untracked custom op
was absent; the supported path setting reused the existing local build without
changing code or installing packages. Its SHA-256 is
`2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
Deprecation warnings are preserved in the local logs. Verbatim author source
retains its original whitespace to preserve source hashes.

Exact test modules are listed in local logs/commands for this integration:
`/tmp/bayesfilter-walnutpie-commit-tests-20261030.log`,
`/tmp/bayesfilter-walnutpie-integration-tests-20261030-retry.log`,
`/tmp/bayesfilter-walnutpie-premerge-sir-tests-20261030.log`, and
`/tmp/bayesfilter-walnutpie-sir-repair-tests-20261030.log`.
The manuscript command is
`latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bayesfilter-walnutpie-merge-pdf-20261030 modern_hmc_survey.tex`,
run from the integration checkout's `docs` directory.

These checks support this integration and the localized frozen-input repair.
They do not represent a full repository test suite or new performance/posterior
evidence. Existing benchmark results retain their original environments and
source hashes.
