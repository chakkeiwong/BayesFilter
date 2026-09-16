"""A10 post-run diagnostic reporting only; no runtime or selection decisions."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'docs/benchmarks/artifacts/observation_tt_robust_guide_20260916'
ARM_NAMES = ('baseline', 'guide', 'stable', 'full', 'transition',
             'stationary_prior', 'sgqf_gaussian', 'sgqf_joint')


def read(path):
    return json.loads(path.read_text())


def eligible(entry, name):
    m = entry['metrics'].get(name)
    return bool(entry['reference_pass'] and m and m['log_evidence_screen']['passed']
                and not m['cdf_bracket_failures'] and not m['consumer_invalid_steps'])


def mean(values):
    return statistics.mean(values) if values else None


def main():
    global OUT
    attempt = sys.argv[1] if len(sys.argv)>1 else 'attempt-confirmation-01'
    assert attempt in ('attempt-confirmation-01','attempt-confirmation-02')
    folder = RUNS / attempt
    if attempt != 'attempt-confirmation-01':
        OUT = OUT / attempt
        OUT.mkdir(exist_ok=True)
    manifest = read(folder / 'run_manifest.json')
    assert manifest['status'] == 'COMPLETE'
    result = read(folder / 'result.json')
    entries = [read(f) for f in sorted(folder.glob('*/summary.json'))]
    assert len(entries) == 24
    provenance = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
                  for name, digest in manifest['source_hashes'].items()}
    assert all(provenance.values()), 'Source drift during or after confirmation'
    for name, digest in manifest['source_hashes'].items():
        original = Path(name)
        archived = OUT / 'source-snapshot' / original.relative_to(ROOT)
        if archived.exists():
            assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
        else:
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, archived)
    controls = read(folder / 'selected-controls.json')
    summary = dict(status='COMPLETE', source_hashes_unchanged=provenance,
                   controls=controls['controls'], rows=[], sequence_failures=[],
                   guide_methods={}, reference_failures=[], chart_minima={},
                   inference=result.get('inference'), timing_scope='Observed execution order, compilation included; descriptive only')
    for d in (1, 4):
        rows = [e for e in entries if e['dimension'] == d]
        assert len(rows) == 12
        methods = {}
        for e in rows:
            for method, count in e['guide_methods'].items():
                methods[method] = methods.get(method, 0) + count
            if e['failures']:
                summary['sequence_failures'].append(dict(dimension=d, sequence=e['sequence'], failures=e['failures']))
            if not e['reference_pass']:
                summary['reference_failures'].append(dict(dimension=d, sequence=e['sequence'], reference=e['reference']))
        summary['guide_methods'][str(d)] = methods
        common = [e for e in rows if all(eligible(e, n) for n in ARM_NAMES[:4])]
        for name in ARM_NAMES:
            available = [e for e in rows if name in e['metrics']]
            valid = [e for e in rows if eligible(e, name)]
            metrics = [e['metrics'][name] for e in available]
            times = []
            for e in available:
                t = e['times'][name]
                guide = (e['original_guide_seconds'] if name == 'baseline' else
                         e['robust_guide_seconds'] if name not in ('transition', 'stationary_prior') else 0.)
                build = t['build_seconds'] + (e['times']['stable']['build_seconds'] if name == 'full' else 0.)
                particle = t['particle_seconds'] / t['particle_replicates']
                times.append((guide, build, particle, guide + build + particle))
            summary['rows'].append(dict(dimension=d, method=name, completed=len(available),
                eligible=len(valid), common_valid_tt_sequences=len(common),
                mse_all_completed=mean([m['regimes']['all']['mse'] for m in metrics]),
                mse_valid=mean([e['metrics'][name]['regimes']['all']['mse'] for e in valid]),
                mse_common_tt=mean([e['metrics'][name]['regimes']['all']['mse'] for e in common if eligible(e, name)]),
                ess_mean=mean([m['particle_diagnostics']['ess']['mean'] for m in metrics]),
                ess_minimum=min([m['particle_diagnostics']['ess']['minimum'] for m in metrics], default=None),
                largest_weight=max([m['particle_diagnostics']['maximum_weight']['maximum'] for m in metrics], default=None),
                resampling_rate=mean([m['particle_diagnostics']['resampling_rate'] for m in metrics]),
                maximum_absolute_log_evidence_bias=max([abs(m['log_evidence_bias']) for m in metrics], default=None),
                log_evidence_vetoes=sum(not m['log_evidence_screen']['passed'] for m in metrics),
                cdf_failures=sum(m['cdf_bracket_failures'] for m in metrics),
                nonfinite_steps=sum(m['consumer_invalid_steps'] for m in metrics),
                guide_seconds=mean([t[0] for t in times]), build_seconds=mean([t[1] for t in times]),
                particle_seconds=mean([t[2] for t in times]), total_seconds_one_run=mean([t[3] for t in times])))
        for name in ARM_NAMES[:3]:
            values=[]
            for f in folder.glob(f'd{d}-*/{name}/path.json'):
                values.extend(read(f)['chart_covariance_minimum'])
            summary['chart_minima'][f'd{d}-{name}'] = min(values) if values else None
    # Preserve the driver's predeclared inference without recomputing or selecting.
    if summary['inference'] is None:
        candidates = [f for f in folder.glob('*.json') if 'inference' in f.name]
        assert len(candidates) == 1, 'Expected one predeclared inference artifact'
        summary['inference'] = read(candidates[0])
    (OUT / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# A10 fresh confirmation comparison', '',
             'All continuous aggregates below are descriptive. MSE is normalized filtering-mean error against the independent reference. Failed/missing cases remain visible.', '',
             '| d | Method | Completed / eligible | MSE (completed) | ESS mean / minimum | Guide + build s | Particle s |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summary['rows']:
        if r['completed']:
            lines.append(f"| {r['dimension']} | {r['method']} | {r['completed']}/12 / {r['eligible']}/12 | {r['mse_all_completed']:.7g} | {r['ess_mean']:.2f} / {r['ess_minimum']:.2f} | {r['guide_seconds'] + r['build_seconds']:.3f} | {r['particle_seconds']:.3f} |")
    lines += ['', 'Particle time is per N512, T20 repetition. Build time is per fitted path and includes the appropriate guide. The full mixture reuses the stable TT fit; its standalone cost includes that fit. Compilation and fixed execution order are included, so this is not a randomized runtime benchmark.', '',
              'Machine-readable coverage, numerical checks, unmodified inference, controls, guide routes, reference failures and source-integrity checks: comparison.json.', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(dict(sequences=len(entries), reference_failures=len(summary['reference_failures']),
                          failed_sequences=len(summary['sequence_failures']), output=str(OUT / 'comparison.json'))))


if __name__ == '__main__':
    main()
