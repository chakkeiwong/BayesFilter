"""Independent standard-library checks of the completed A09 evidence files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirmation',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.confirmation);cal=Path(args.calibration);out=Path(args.output)
    result=read(root/'result.json');manifest=read(root/'run_manifest.json')
    controls=read(cal/'selected-controls.json')
    assert result['status']==manifest['status']=='COMPLETE'
    assert not (root/'invalidation.json').exists()
    assert manifest['source_unchanged'] and manifest['seeds']['revision']==2
    assert len(result['sequences'])==24
    methods=('baseline','capacity','preservation','tt_pair_block','transition','stationary_prior','sgqf_gaussian','sgqf_joint')
    all_streams=set();records=0;cdf_checks=0;reference_records=0;failures=[];coverage=[]
    expected_records=0;expected_cdf=0;fits_checked=0
    for entry in result['sequences']:
        d=entry['dimension'];s=entry['sequence'];seed=entry['data_seed']
        assert seed==100000000+10000000*(24*manifest['seeds']['partition']+12*(d==4)+s)
        dest=root/f'd{d}-s{s:02d}'
        assert read(dest/'data.json')['seed']==seed
        missing=sorted(set(methods)-set(entry['metrics']))
        if missing or not entry['reference_pass'] or entry['failures']:
            coverage.append(dict(dimension=d,sequence=s,missing=missing,reference_pass=entry['reference_pass'],failures=entry['failures']))
        assert set(entry['metrics']).issubset(methods)
        unit_streams=set()
        for name in methods:
            if name not in entry['metrics']: continue
            if name in ('baseline','capacity','preservation'):
                for t in range(1,20):
                    fitted=read(dest/name/f'fit-t{t}.json')
                    assert fitted['config']==controls['dimensions'][str(d)]['selected'][name]
                    assert fitted['row_diagnostics']['train']['seed']==seed+2000000+t
                    assert fitted['row_diagnostics']['validation']['seed']==seed+2100000+t
                    fits_checked+=1
            expected_records+=80
            if name in ('baseline','capacity','preservation','tt_pair_block'): expected_cdf+=76
            for repetition in range(4):
                particle=read(dest/name/f'particles-r{repetition}.json')
                expected=seed+3000000+1000*repetition
                assert particle['seed']==expected and particle['count']==512
                assert len(particle['steps'])==20
                if name==next(n for n in methods if n in entry['metrics']):
                    stream=set(range(expected,expected+20))
                    assert not unit_streams.intersection(stream)
                    unit_streams.update(stream)
                for t,step in enumerate(particle['steps']):
                    assert step['time']==t
                    assert all(math.isfinite(value) for value in numbers(step))
                    if 'finite' in step: assert step['finite']
                    assert 0<step['ess']<=512*(1+1e-12)
                    assert 0<step['maximum_weight']<=1
                    if name in ('baseline','capacity','preservation','tt_pair_block') and t>0:
                        assert step['finite']
                        assert step['cdf_bracket_valid']
                        assert step['cdf_residual']<1e-10
                        cdf_checks+=1
                    records+=1
            if not entry['metrics'][name]['log_evidence_screen']['passed']:
                failures.append(dict(dimension=d,sequence=s,method=name))
        if d==4:
            for level,particle_count in enumerate((32768,65536,131072)):
                path=dest/f'reference-{particle_count}.json'
                if not path.exists(): continue
                ref=read(path)
                for repetition,particle in enumerate(ref['runs']):
                    expected=seed+4000000+1000000*level+1000*repetition
                    assert particle['seed']==expected
                    stream=set(range(expected,expected+20))
                    assert not unit_streams.intersection(stream)
                    unit_streams.update(stream)
                    reference_records+=len(particle['steps'])
        assert not all_streams.intersection(unit_streams)
        all_streams.update(unit_streams)
    assert records==expected_records and cdf_checks==expected_cdf
    finite_values=0
    for path in root.rglob('*.json'):
        if 'executed-sources' in path.parts: continue
        for value in numbers(read(path)):
            assert math.isfinite(value), str(path)
            finite_values+=1
    for contrast in result['inference']['primary_contrasts']:
        values=[entry['metrics'][contrast['candidate']]['regimes']['all']['mse']-entry['metrics']['baseline']['regimes']['all']['mse']
                for entry in result['sequences'] if entry['dimension']==contrast['dimension'] and entry['reference_pass']
                and all(n in entry['metrics'] for n in ('baseline',contrast['candidate']))]
        if not values:
            assert contrast['mean_delta'] is None
            continue
        center=mean(values)
        se=math.sqrt(sum((v-center)**2 for v in values)/(len(values)*(len(values)-1))) if len(values)>1 else 0.
        assert abs(center-contrast['mean_delta'])<1e-15
        assert abs(se-contrast['se'])<1e-15
        if not contrast['interval_eligible']:
            assert contrast['upper'] is None and contrast['lower'] is None
            continue
        width=result['inference']['simultaneous_critical']*se
        assert abs(contrast['upper']-center-width)<1e-15
        assert abs(contrast['lower']-center+width)<1e-15
    closure=0
    for directory in (cal,root):
        m=read(directory/'run_manifest.json')
        for source,sha in m['source_hashes'].items():
            path=directory/'executed-sources'/Path(source).relative_to(Path.cwd())
            assert digest(path)==sha,str(path)
            closure+=1
    audits=list(cal.glob('d*-s*/same-target-audit/*.json'))
    assert len(audits)==54
    improved=sum(read(path)['audit']['fitted']['defended_h2']<read(path)['audit']['initial']['defended_h2'] for path in audits)
    output=dict(status='PASS',confirmation=str(root),particle_time_records=records,
                checked_tt_cdf_records=cdf_checks,reference_particle_time_records=reference_records,
                finite_numeric_values=finite_values,distinct_particle_reference_scalar_seeds=len(all_streams),
                source_snapshots_verified=closure,frozen_configs_and_fit_seeds_verified=fits_checked,same_target_audits=54,
                fitted_h2_below_initializer=improved,evidence_screen_failures=failures,coverage_failures=coverage,
                primary_contrast_arithmetic='independently recomputed; bootstrap critical value retained from frozen driver',
                reviewer='executor self-audit; no independent reviewer claimed')
    (out/'audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:output[key] for key in ('status','particle_time_records','checked_tt_cdf_records',
        'source_snapshots_verified','frozen_configs_and_fit_seeds_verified','same_target_audits')}))


if __name__=='__main__':
    main()
