"""Post-run diagnostic inspection only; never imported by a runtime candidate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / 'attempt-confirmation-02'
RUNS = ROOT / 'docs/benchmarks/artifacts/observation_tt_robust_guide_20260916'


def read(path):
    return json.loads(path.read_text())


def main():
    folder = RUNS / 'attempt-confirmation-02'
    manifest = read(folder / 'run_manifest.json')
    assert manifest['status'] == 'COMPLETE'
    result = read(folder / 'result.json')
    assert len(result['sequences']) == 24
    snapshot = OUT / 'source-snapshot'
    source_checks = {
        path: hashlib.sha256((snapshot / Path(path).relative_to(ROOT)).read_bytes()).hexdigest() == digest
        for path, digest in manifest['source_hashes'].items()
    }
    assert all(source_checks.values())
    negative, margins, fractions, fallback = [], [], [], []
    selected_reference_counts = {}
    for sequence in sorted(folder.glob('d*-s*')):
        info = read(sequence / 'reference.json')
        if 'selected_N' in info:
            key = str(info['selected_N'])
            selected_reference_counts[key] = selected_reference_counts.get(key, 0) + 1
        for t, step in enumerate(read(sequence / 'robust-guide.json')):
            if step['fallback']:
                fallback.append(dict(sequence=sequence.name, time=t, method=step['method']))
        for path in sequence.glob('*/fit-*.json'):
            fit = read(path)
            if fit['initial_scale'] < 0:
                negative.append(dict(path=str(path.relative_to(folder)), scale=fit['initial_scale']))
        for path in sequence.glob('full/particles-r*.json'):
            for step in read(path)['steps']:
                margins.append(step['log_density_lower_bound_margin'])
                fractions.append(step['physical_defense_fraction'])
    seeds = {}
    for attempt in ('attempt-calibration-02', 'attempt-confirmation-01', 'attempt-confirmation-02'):
        seeds[attempt] = [e['data_seed'] for e in read(RUNS / attempt / 'result.json')['sequences']]
    all_seeds = [s for group in seeds.values() for s in group]
    assert len(all_seeds) == len(set(all_seeds))
    report = dict(
        status='COMPLETE', classification='post_run_diagnostic_only',
        snapshot_hashes_match=source_checks,
        calibration_selection_identical=read(RUNS / 'attempt-calibration-01/selected-controls.json') == read(RUNS / 'attempt-calibration-02/selected-controls.json'),
        sequence_seeds_disjoint=True, sequence_counts={k: len(v) for k, v in seeds.items()},
        negative_initial_scales=negative, unresolved_guide_fallbacks=fallback,
        reference_selected_counts=selected_reference_counts,
        physical_mixture_steps=len(margins), minimum_density_bound_margin=min(margins),
        mean_physical_draw_fraction=sum(fractions)/len(fractions),
        physical_density_bound_check=min(margins) >= -1e-12,
        nonclaim='Finite checks do not certify quadrature accuracy, realized ESS, global derivatives or default readiness.',
    )
    (OUT / 'terminal-checks.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}))
    print(json.dumps(dict(negative_scales=len(negative), unresolved_fallbacks=len(fallback), reference_counts=selected_reference_counts)))


if __name__ == '__main__':
    main()
