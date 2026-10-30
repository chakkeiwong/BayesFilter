#!/usr/bin/env python3
"""Independent diagnostic digitization of the published PDF's vector ESS plots.

No fitted author arrays are available. Values retain PDF rendering precision only.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / '.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf'
SPECS = [
    dict(page=39, figure=15, model='sir', transform='matrix(1,0,0,-1,193.366,571.021)',
         x0=0.0011875, xmax=227.610562, last_time=20,
         ylow=19.575688, yhigh=97.872563, flow=0.2, fhigh=1.0,
         colors={'0%,0%,100%': 'rank40', '75%,50%,25%': 'rank20', '100%,0%,0%': 'rank10'}),
    dict(page=41, figure=17, model='pp', transform='matrix(1,0,0,-1,230.265,455.45)',
         x0=0.000625, xmax=168.164687, last_time=10,
         ylow=0.00078125, yhigh=97.871875, flow=0.0, fhigh=1.0,
         colors={'100%,0%,0%': 'nonlinear', '0%,0%,100%': 'linear'}),
]

def extract(svg, spec):
    paths = list(ET.parse(svg).getroot().iter('{http://www.w3.org/2000/svg}path'))
    rows = []
    for color, method in spec['colors'].items():
        lines = []
        for path in paths:
            if path.get('transform') != spec['transform'] or 'stroke:rgb(' + color + ')' not in path.get('style', ''):
                continue
            d = path.get('d', '')
            if re.search('[A-KN-Zac-z]', d):
                continue  # Closed markers and Bezier glyphs are not data polylines.
            numbers = list(map(float, re.findall(r'-?\d+(?:\.\d*)?(?:e[+-]?\d+)?', d)))
            if len(numbers) % 2:
                raise ValueError('Odd coordinate count')
            lines.append(list(zip(numbers[::2], numbers[1::2])))
        medians = [line for line in lines if len(line) == spec['last_time']]
        if len(medians) != 1:
            raise ValueError(f'Expected one median curve for {method}, found {len(medians)}')
        for index, (x, y) in enumerate(medians[0], 1):
            t = (x - spec['x0']) / (spec['xmax'] - spec['x0']) * spec['last_time']
            if abs(t - index) > 0.001:
                raise ValueError('Plotted time calibration mismatch')
            bars = [b for b in lines if len(b) == 2 and all(abs(bx - x) < 1e-5 for bx, _ in b)]
            if len(bars) != 2 or any(abs(b[0][1] - y) > 1e-5 for b in bars):
                raise ValueError('Expected two quartile segments anchored to median')
            low, high = sorted(b[1][1] for b in bars)
            if not low <= y <= high:
                raise ValueError('Quartiles do not bracket median')
            def fraction(v):
                return spec['flow'] + (v - spec['ylow']) / (spec['yhigh'] - spec['ylow']) * (spec['fhigh'] - spec['flow'])
            rows.append(dict(model=spec['model'], method=method, figure=spec['figure'], time=index,
                             q25=fraction(low), median=fraction(y), q75=fraction(high)))
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True, type=Path)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for spec in SPECS:
        svg = args.output_root / f'paper-page{spec["page"]}.svg'
        subprocess.run(['pdftocairo', '-svg', '-f', str(spec['page']), '-l', str(spec['page']), str(PAPER), str(svg)], check=True)
        rows.extend(extract(svg, spec))
    with (args.output_root / 'published-ess.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(source_url='https://jmlr.org/papers/v25/23-0743.html', paper_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest(),
                    interpretation='PDF vector digitization; approximate plotted quartiles, not original experiment arrays',
                    quartile_source='Section 6 introduction: 25%, 50%, 75% ESS quantiles from 40 repeated experiments',
                    calibration=SPECS, rows=len(rows),
                    discrepancy='Figure 17 has ten points at times 1..10 and nonlinear median about 80.7% at time 10. Section 6.4 separately states about 40% after 20 steps. The plot does not show time 20. No rescaling or extrapolation applied.',
                    resolution='Coordinates rounded by PDF rendering. Report percentages to one decimal place; no statistical uncertainty inferred from digitization.')
    (args.output_root / 'source-and-calibration.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'rows':len(rows), 'terminal': [r for r in rows if r['time'] == (20 if r['model']=='sir' else 10)]}, indent=2))

if __name__ == '__main__':
    main()
