"""Diagnostic-only verification of the monograph's saved result snapshot.

Checks file identity and reported terminal decisions without loading a model,
running a sampler, or treating saved summaries as an independent replication.
"""
from pathlib import Path
import hashlib
import json
import math


def main():
    root = Path(__file__).with_name('stan-walnutpie-evidence-20261031')
    read = lambda name: json.loads((root / name).read_text())
    records = read('index.json')['records']
    for record in records:
        data = (root / record['snapshot']).read_bytes()
        assert len(data) == record['bytes'], record['snapshot']
        assert hashlib.sha256(data).hexdigest() == record['sha256'], record['snapshot']
    full = read('full80/result.json')
    assert full['posterior_passed'] and full['comparison_complete']
    assert (full['adaptation_count'], full['retained_count']) == (105, 1400)
    candidate = read('full80/candidate.json')
    expected = dict(max_rhat=1.0055516436462204,
                    min_bulk_ess=1435.833895763604,
                    min_tail_ess=953.4339446758111,
                    max_controller_mcse_sd=0.027746152977120988)
    for key, value in expected.items():
        assert math.isclose(candidate[key], value, rel_tol=1e-12), key
    comparison = read('full80/comparison.json')
    assert comparison['qualified'] and not comparison['flagged_parameters']
    assert comparison['parameters'] == 80
    review = read('full80-review/checks.json')
    assert review['passed'] and review['sample_and_checkpoint_files_verified'] == 2409
    assert review['current_model_source_drift']
    assert not review['macrofinance_model_evaluated']
    million = read('million/comparison.json')
    assert len(million['cases']) == 4 and not million['missing']
    for case in million['cases']:
        d = case['diagnostics']
        assert d['draws'] == 1000000 and d['discard'] == 0
        assert d['posterior_screen']
    flags = {(c['arm'], c['quantity']) for c in million['comparisons']
             if c.get('eligible') and c.get('detected_difference')}
    assert flags == {('author-original', 'z0'), ('author-original', 'tail-3'),
                     ('author-original', 'tail-6'), ('author-corrected', 'z2')}
    assert million['bonferroni_family_size'] == 91
    print(json.dumps(dict(verified_files=len(records), full80=expected,
        million=[dict(arm=c['arm'],
            max_rhat=c['diagnostics']['max_rhat'],
            min_bulk_ess=c['diagnostics']['min_bulk_ess'],
            min_tail_ess=c['diagnostics']['min_tail_ess'],
            sampling_seconds=c['sampling_seconds']) for c in million['cases']],
        eligible_discrepancies=sorted(flags),
        interpretation='Saved-result verification; no new sampling or target replay.'),
        indent=2))


if __name__ == '__main__':
    main()
