"""Read-only reconciliation of historical full fits; not new confirmation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(args.repo))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    base = args.repo / 'docs/plans/artifacts/hmc-repair-master-2026-09-16'
    authority = read(base / 'm25-r1/public-final-audit-r2/result.json')['terminal_public_groups']
    rows, summaries = [], {}
    for target, planned in (('gaussian',128),('beta_binomial',128),('rotated_gaussian',8),('lgssm_location',8)):
        label = 'm21-merged-confirmation-' + target
        root = base / 'm21-r2/public-confirmation-cpu-r1' / target / label
        design = read(root/'design.json')
        counts = Counter()
        intervals = {}
        for index in range(planned):
            fit = root / f'replication-{index:04d}'
            pipe = read(fit/'pipeline.json')
            assessment = read(fit/'independent_assessment.json')
            selected = pipe['selection']['candidate_ids']
            members = [m for m in pipe['members'] if m['candidate_id'] in selected]
            assert len(members) <= 1
            member = members[0] if members else {}
            post = member.get('posterior', {})
            audit_member = next((m for m in assessment['members']
                                 if m['candidate_id'] in selected), {})
            pair = audit_member.get('stopping_pair', {})
            delivered = post.get('passed') is True
            categories = []
            if not members:
                categories.append('no_selected_member')
            if post.get('warmup_cap_hit'):
                categories.append('warmup_cap')
            if post.get('retained_cap_hit'):
                categories.append('retained_cap')
            if post.get('hard_vetoes'):
                categories.append('numerical_health_veto')
            if delivered:
                categories.append('delivered')
            if not categories:
                categories.append('other_incomplete')
            counts.update(categories)
            for arm in ('stopped','fixed'):
                for name, interval in pair.get(arm,{}).items():
                    acc = intervals.setdefault(arm+':'+name, Counter())
                    acc['available'] += bool(interval['available'])
                    acc['covered'] += bool(interval['covered'])
                    acc['qualified_and_covered'] += bool(delivered and interval['covered'])
            config = post.get('config', {})
            row = {'target': target, 'replication': index, 'pipeline': str(fit/'pipeline.json'),
                   'design_file': str(root/'design.json'), 'source_binding': pipe['source_binding'],
                   'data':pipe['data'], 'numerical_route':pipe['numerical_route'],
                   'selection':pipe['selection'], 'verified_candidate_ids':pipe['verified_candidate_ids'],
                   'unassessed_siblings':sum(m['status']=='unassessed_by_design' for m in pipe['members']),
                   'completion':pipe['completion'], 'categories':categories, 'delivered':delivered,
                   'member': {k:member.get(k) for k in ('candidate_id','L','epsilon','draws_path','warmup_path')},
                   'posterior_policy':config, 'counts':{k:post.get(k) for k in (
                       'warmup_results_per_chain','retained_results_per_chain')},
                   'hard_vetoes':post.get('hard_vetoes',[]), 'stopping_pair':pair,
                   'last_warmup_check':post.get('warmup_checks',[None])[-1],
                   'last_retained_check':(post.get('retained_checks') or [None])[-1]}
            rows.append(row)
        expected=authority[label]
        assert counts['delivered']==expected['delivered']
        assert counts['warmup_cap']==expected['warmup_caps']
        assert counts['retained_cap']==expected['retained_caps']
        for name, report in expected['stopped_versus_fixed']['quantities'].items():
            for arm, values in report['arms'].items():
                observed=intervals.get(arm+':'+name, Counter())
                for metric in ('available','covered'):
                    assert observed[metric]==values[metric],(target,arm,name,metric)
        summaries[target]={'planned':planned,'recorded':sum(r['target']==target for r in rows),
                           'categories':dict(counts),'intervals':intervals,
                           'matches_M25_authority':True,'design':design}
    assert len(rows)==272
    result={'schema':'bayesfilter.hmc_historical_failure_inventory.v1','rows':rows,'summary':summaries,
            'role':'development reanalysis of historical evidence; no new fits or promotion',
            'units':'independently executed historical full fits; members not additional replications',
            'source_sha256':sources,'elapsed_seconds':time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({'output':str(args.output),'fits':len(rows),
        'summary':{k:v['categories'] for k,v in summaries.items()},'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':
    main()
