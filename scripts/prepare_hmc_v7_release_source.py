"""Assemble a fresh HMC validation tree from Git plus explicit scoped overlays.

No checkout, index or unrelated worktree content is changed. The manifest lists
every copied file and separates the Git baseline from intentional overlays.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build-custom-op',action='store_true',
                        help='Build the native TensorFlow dependency in this fresh tree')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source = root/'source'
    source.mkdir()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    archive = subprocess.check_output(['git','archive','--format=tar',revision,
        'bayesfilter','tests','pytest.ini','CMakeLists.txt','custom_call_status_stub.cc','README.md'],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source,filter='data')
    paths = set()
    for pattern in ('bayesfilter/inference/hmc*.py','bayesfilter/testing/acceptance*.py',
                    'bayesfilter/testing/inference_validation/**/*.py','tests/test_hmc*.py',
                    'tests/inference_validation/**/*.py'):
        paths.update(p.relative_to(repo).as_posix() for p in repo.glob(pattern))
    paths.update((
        'bayesfilter/inference/__init__.py',
        'bayesfilter/inference/tuning_contract.py','bayesfilter/runtime/execution_budget.py',
        'bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py',
        'bayesfilter/testing/lgssm_generic_target_adapter_tf.py',
        'bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py',
        'bayesfilter/testing/neutra_model_registry_tf.py','bayesfilter/testing/tf_hmc_readiness.py',
        'docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-result-2026-07-08.md',
        'docs/plans/bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md',
        'docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json',
        'docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/fixture_T120_seed20260709_301.json',
        'docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_contract_v1.json',
        'docs/validation/hmc-acceptance-integration-tests.txt',
        'tests/data/hmc_endpoint_return_trial.json',
        'tests/data/hmc_health_source08_reference.txt',
        'scripts/run_hmc_v7_release_prices.py',
        'scripts/analyze_hmc_v7_confirmation.py',
        'scripts/run_hmc_v7_confirmation.py',
        'scripts/render_hmc_tuning_interface_docs.py',
        'scripts/inventory_hmc_tuning_routes.py',
        'AGENTS.md',
        'docs/main.tex',
        'docs/reference/hmc-tuning-interface.md',
        'docs/chapters/ch21b_hmc_tuning_interfaces.tex',
        'docs/generated/hmc_tuning_route_table.md',
        'docs/generated/hmc_tuning_route_table.tex',
        'docs/examples/hmc_tuning_route_selection.py',
        'docs/examples/hmc_tuning_ordinary.py',
        'docs/examples/hmc_tuning_covariance_first.py',
        'docs/examples/hmc_tuning_neural_force_binding.py',
        'docs/examples/hmc_tuning_fixed_transport.py',
    ))
    overlays = {}
    for name in sorted(paths):
        original = repo/name
        if not original.is_file():
            raise FileNotFoundError(original)
        destination = source/name
        raw = original.read_bytes()
        if destination.is_file() and destination.read_bytes() == raw:
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw)
        overlays[name] = hashlib.sha256(raw).hexdigest()
    build = []
    if args.build_custom_op:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true',
                   TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',
                   OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
        commands = [(['cmake','-S',str(source),'-B',str(root/'build'),
                     '-DPython3_EXECUTABLE='+sys.executable],180),
                    (['cmake','--build',str(root/'build'),'--parallel','2'],420)]
        for index,(command,limit) in enumerate(commands):
            started = time.monotonic()
            with (root/f'build-{index}.log').open('x') as log:
                try:
                    code = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                          timeout=limit).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            build.append({'command':command,'exit_code':code,'wall_seconds':time.monotonic()-started,
                          'gpu_intentionally_hidden':True,'memory_growth_environment':'true'})
            (root/'build-receipt.json').write_text(json.dumps(build,indent=2)+'\n')
            if code:
                raise RuntimeError(f'custom op build failed; see {root}/build-{index}.log')
    manifest = {p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.rglob('*')) if p.is_file()}
    (root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'assembly.json').write_text(json.dumps({
        'schema':'bayesfilter.hmc_release_source_assembly.v1','git_commit':revision,
        'overlays':overlays,'source':str(source),'file_count':len(manifest),
        'native_library':'built_from_frozen_source' if build else 'build_required_for_state_space_targets',
        'build_receipt':'build-receipt.json' if build else None,
        'excluded_worktree_changes':'q20 and learned NeuTra runtime/training; Git baseline retained',
        'scope':'scoped validation tree, not a package/publication or a claim that Git contains the overlays',
        'plan':'docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md'},indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'overlays':len(overlays),'source':str(source)}))


if __name__ == '__main__':
    main()
