"""Build the standalone native diagnostic bridge; never installs packages."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
BUILD = ROOT / ".localresources/native_sampler_bridge"
STAN = ROOT / ".localresources/cmdstan/cmdstan-2.37.0"
WALNUT = ROOT / ".localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    commands = []
    def run(cmd, name):
        start = time.monotonic()
        with (BUILD / (name + ".log")).open("w") as log:
            result = subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT)
        commands.append({"command": [str(x) for x in cmd], "seconds": time.monotonic()-start, "returncode": result.returncode})
        (BUILD / "build_manifest.json").write_text(json.dumps(commands, indent=2))
        if result.returncode:
            print((BUILD/(name+".log")).read_text()[-9000:])
            raise SystemExit(result.returncode)
    run([STAN/"bin/stanc", "--name=native_targets", "--o="+str(BUILD/"native_targets.hpp"), ROOT/"docs/benchmarks/stan_targets.stan"], "stanc")
    math = STAN / "stan/lib/stan_math"
    includes = [STAN/"stan/src", math, math/"lib/eigen_3.4.0", math/"lib/boost_1.87.0", math/"lib/tbb_2020.3/include", math/"lib/sundials_6.1.1/include", WALNUT/"include", BUILD]
    native = ["g++", "-std=c++20", "-O3", "-fPIC", "-shared", "-pthread", "-D_REENTRANT", "-DBOOST_DISABLE_ASSERTS", "-DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION"]
    native += ["-I"+str(p) for p in includes]
    native += [SOURCE/"native.cc", "-L"+str(math/"lib/tbb"), "-Wl,-rpath,"+str(math/"lib/tbb"), "-ltbb", "-o", BUILD/"libbf_native.so"]
    run(native, "native")
    run(["g++","-O3","-std=c++20","-pthread","-I"+str(math/"lib/eigen_3.4.0"),
         "-I"+str(WALNUT/"include"),SOURCE/"walnut_oracle.cc","-o",BUILD/"walnut_oracle"],"walnut_oracle")
    if not args.native_only:
        import tensorflow as tf
        inc, lib = Path(tf.sysconfig.get_include()), Path(tf.sysconfig.get_lib())
        prefix = Path(os.sys.prefix)
        extra = [inc/"external/llvm-project/mlir/include", inc/"external/llvm-project/llvm/include", prefix/"include", inc/"external/highwayhash", inc/"external/farmhash_archive/src", inc/"third_party/gpus/cuda/include"]
        op = ["g++", "-O2", "-fPIC", "-shared", "-pthread"] + tf.sysconfig.get_compile_flags()
        op += ["-I"+str(p) for p in extra if p.exists()]
        op += [SOURCE/"op.cc", "-L"+str(BUILD), "-lbf_native", str(lib/"libtensorflow_cc.so.2"), str(lib/"libtensorflow_framework.so.2"), str(prefix/"lib/libcudart.so.12"), "-Wl,-rpath,"+str(BUILD)+":"+str(lib)+":"+str(prefix/"lib"), "-o", BUILD/"_native_sampler_ops.so"]
        run(op, "op")
    sources=[*SOURCE.glob("*.cc"),*SOURCE.glob("*.h"),*SOURCE.glob("*.py"),
             ROOT/"docs/benchmarks/stan_targets.stan",BUILD/"native_targets.hpp",BUILD/"libbf_native.so",
             *WALNUT.glob("include/**/*.hpp"),
             STAN/"stan/src/stan/services/sample/hmc_nuts_diag_e_adapt.hpp"]
    if not args.native_only:sources.append(BUILD/"_native_sampler_ops.so")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (BUILD/"source_hashes.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps({"build":str(BUILD),"commands":commands},indent=2))

if __name__ == "__main__":
    main()
