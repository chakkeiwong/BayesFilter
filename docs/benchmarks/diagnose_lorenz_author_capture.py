"""CPU independent-reference capture and MAT/binary export; never a runtime.

NumPy/SciPy operations are confined to this explicitly diagnostic reference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.io import loadmat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = args.output/'sources/cSMC_Resample.m'
    baseline_summary = json.loads((args.reference/'summary.json').read_text())
    assert baseline_summary['status'] == 'PASS'
    for name in ('cSMC_Resample.m','cSMC_VaryF.m','RungeKutta4.m',
                 'Systematic_Resampling.m','SimulatedData.mat'):
        assert sha(args.output/'sources'/name) == sha(args.reference/'sources'/name)
    for key,expected in baseline_summary['dependency_sha256'].items():
        assert sha(Path(baseline_summary[key])) == expected
    original = source.read_text()
    anchor = "            BetaHat = (DesignMatrix' * DesignMatrix) \\ (DesignMatrix' * Value); % (d^2/2 + 3d/2 + 1) x 1"
    assert original.count(anchor) == 1
    addition = ("\n            global LORENZ_AUDIT_FITS;\n"
                "            LORENZ_AUDIT_FITS{end+1} = struct('time',t,'X',squeeze(SMC.Trajectory(:,:,t)),"
                "'Value',Value,'BetaHat',BetaHat,'design_edges',DesignMatrix([1,end],:));")
    modified = original.replace(anchor, anchor+addition)
    instrumented = args.output/'instrumented'
    instrumented.mkdir()
    (instrumented/source.name).write_text(modified)
    (args.output/'instrumentation.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), modified.splitlines(True), fromfile='original', tofile='capture_only')))
    command = ['octave', '--no-gui', '--quiet',
               str(args.output/'sources/reference_lorenz_author_capture.m'),
               str(Path.cwd()), str(args.output), str(args.reference)]
    subprocess.run(command, check=True)
    index = dict(schema='lorenz_author_capture_v1', cases=[], arrays={},
                 source_sha256=sha(source), instrumented_sha256=sha(instrumented/source.name),
                 reference_root=str(args.reference), dtype='float64', byte_order='little')

    def save(name, value):
        assert np.isrealobj(value)
        array = np.ascontiguousarray(value, dtype='<f8')
        assert np.isrealobj(array) and np.all(np.isfinite(array))
        path = args.output/(name+'.bin')
        path.write_bytes(array.tobytes())
        index['arrays'][name] = dict(file=path.name, shape=list(array.shape), sha256=sha(path))

    for forcing in (2.5, 4.8801, 8.5):
        for seed in (20260929, 20260930):
            filename = 'author-F%.4f-seed%d.mat' % (forcing, seed)
            capture = loadmat(args.output/filename, simplify_cells=True)
            source_data = loadmat(args.reference/filename, simplify_cells=True)
            assert capture['exact_source_replay'] == 1
            records = [item if isinstance(item, dict) else vars(item)
                       for item in capture['captured']]
            result = source_data['result']
            count = int(capture['refinements'])
            assert result.shape == (count+1, 2) and len(records) == 101*count
            case_id = len(index['cases'])
            save('case%d_y' % case_id, source_data['y'])
            for stage in range(count+1):
                prefix = 'case%d_stage%d_' % (case_id, stage)
                policy, smc = result[stage]
                policy = policy if isinstance(policy, dict) else vars(policy)
                smc = smc if isinstance(smc, dict) else vars(smc)
                for key in ('A', 'b', 'c'):
                    save(prefix+key, policy[key])
                save(prefix+'x', np.transpose(smc['Trajectory'], (2, 0, 1)))
                save(prefix+'lw', smc['LogIncrementalWeight'].T)
                save(prefix+'logZ', smc['LogNormConst'])
                save(prefix+'ess', smc['ESS'])
                if stage < count:
                    local = records[stage*101:(stage+1)*101][::-1]
                    assert all(int(item['time']) == t+1 for t,item in enumerate(local))
                    assert np.array_equal(np.stack([item['X'] for item in local]),
                                          np.transpose(smc['Trajectory'], (2, 0, 1)))
                    for key in ('Value', 'BetaHat', 'design_edges'):
                        save(prefix+key, np.stack([item[key] for item in local]))
            index['cases'].append(dict(case=case_id, forcing=forcing, seed=seed,
                refinements=count, exact_source_replay=True,
                original_archive_sha256=sha(args.reference/filename),
                capture_archive_sha256=sha(args.output/filename)))
    assert sum(c['refinements'] for c in index['cases']) == 16
    (args.output/'fixture.json').write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
    summary = dict(status='PASS', scope='IDENTICAL_AUTHOR_REGRESSION_INPUT_CAPTURE',
        replication='INCOMPLETE', cpu_only=True, gpu_intentionally_hidden=True,
        numpy_role='INDEPENDENT_REFERENCE_ONLY', source_results_exactly_unchanged=True,
        cases=index['cases'], captured_regressions=1616, command=command,
        algorithm_seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', cases=6, captured_regressions=1616)))


if __name__ == '__main__':
    main()
