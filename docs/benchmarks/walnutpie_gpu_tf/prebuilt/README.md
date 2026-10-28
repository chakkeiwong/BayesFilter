# Prebuilt Walnutpie C++ reference

The generic executable in `linux-x86_64/20261029/oracle` is **249,008 bytes
(243.2 KiB)**. It is the unchanged binary used in the October 29 optimization
campaign. Compatible Linux users can run the C++ reference and supplied
benchmark targets without installing a compiler, Eigen or Walnutpie.

The source archive is about 1 MiB and includes the matching oracle source,
Walnutpie headers at commit `5710b0334ba69cf0caa35fc02a2338b61d756953`,
Eigen 3.4.0 headers and upstream license notices. `manifest.json` records hashes
for every archived source file, the binary, compiler flags and original build
command. This build took 7.33 seconds on the development machine; avoiding the
compiler/dependency setup may save more time than compilation itself.

## Compatibility

- Linux x86-64 baseline instructions; no `-march=native` requirement.
- glibc 2.34 or newer; libstdc++ exporting `GLIBCXX_3.4.29` and
  `CXXABI_1.3.9`; system libm and libgcc_s.
- No TensorFlow, CUDA, TBB, absolute RPATH or RUNPATH dependency.
- Locally checked after relocation and with library-path overrides removed.
  Other Linux installations have not been independently tested. Rebuild for
  older distributions, ARM, macOS or Windows.

## Use without compiling

From the repository root, run the deterministic reference cases:

~~~bash
docs/benchmarks/walnutpie_gpu_tf/prebuilt/linux-x86_64/20261029/oracle
~~~

This writes 528 JSON records to stdout. It is a deterministic correctness
reference, not a posterior experiment. The existing benchmark workers accept
the same executable through `--oracle`. For a CPU reference validation with an
installed TensorFlow environment:

~~~bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
python -m docs.benchmarks.walnutpie_gpu_tf.worker \
  --mode validate --device cpu --execution-layout packed \
  --oracle docs/benchmarks/walnutpie_gpu_tf/prebuilt/linux-x86_64/20261029/oracle \
  --output /tmp/walnutpie-prebuilt-validation
~~~

Choose a fresh output path for each validation. The optimization master also
accepts `--oracle`; it additionally needs the pinned sources and compiler
metadata for its campaign snapshots. Its `build` phase deliberately rebuilds.
The reference executable itself has neither requirement.
The original `master.py` rebuilds the oracle in its `build`, `qualify` and
`all` phases, including a supplied `--oracle` path. Use its `validate` or
`benchmark` phase with the prebuilt; do not use the tracked file as a build
destination.

To verify the package, run `sha256sum -c SHA256SUMS` from
`linux-x86_64/20261029`. The committed packaging validation records the smoke
checks, relocation, source rebuild and binary identity.

## Rebuild for another host

Extract `sources.tar.gz` into an empty directory. From that directory:

~~~bash
g++ -O3 -std=c++20 -pthread -Ieigen -Iwalnutpie/include oracle.cc -o oracle
~~~

This uses unmodified upstream headers. Notices are preserved in the archive
and in `licenses/`. The manifest identifies the original source snapshot even
when a different compiler produces different binary bytes. For a local
architecture-optimized build, add `-march=native` and revalidate; that binary
should be rebuilt on its destination CPU.

The saved executable implements the supplied analytic benchmark targets. It is
not a general shared-library API for arbitrary TensorFlow models. The older
TensorFlow CPU sampler bridge remains a separate local build because it embeds
environment paths and TensorFlow ABI dependencies. The GPU TensorFlow port
still compiles each stable model signature with XLA on the user's machine.
