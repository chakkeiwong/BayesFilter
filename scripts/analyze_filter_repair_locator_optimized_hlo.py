"""Diagnostic content comparison of saved optimized HLO; no admission authority."""

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HEADER = re.compile(r'^(?:ENTRY )?%([^ ]+) \(')
DEFINITION = re.compile(r'^\s*(?:ROOT )?%([^ ]+) = (.*)$')
SYMBOL = re.compile(r'%([A-Za-z0-9_.-]+)')
METADATA = re.compile(r', metadata=\{[^}]*\}')
OPCODE = re.compile(r'\b([a-z][a-z0-9-]*)\(')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def computations(lines):
    """Keep operand order/layout/attributes and callee content; reject unknown syntax."""
    functions, records = {}, []
    record = None
    body, local, opcodes, roots, callees, elisions = [], {}, collections.Counter(), 0, set(), []
    for number, raw in enumerate(lines, 1):
        header = HEADER.match(raw)
        if header:
            assert record is None
            name = header.group(1)
            assert name not in functions
            functions[name] = None
            record = {'name': name, 'line': number, 'entry': raw.startswith('ENTRY ')}
            body, local, opcodes, roots, callees, elisions = [], {}, collections.Counter(), 0, set(), []
        elif record is not None and raw.rstrip() == '}':
            assert roots == 1 and body
            record.update(normalized_sha256=hashlib.sha256('\n'.join(body).encode()).hexdigest(),
                instruction_count=len(body), opcodes=dict(sorted(opcodes.items())),
                callees=sorted(callees), end_line=number, opaque_constant_lines=elisions,
                constants_fully_printed=not elisions)
            functions[record['name']] = record['normalized_sha256']
            records.append(record)
            record = None
        elif record is not None and raw.strip():
            match = DEFINITION.match(raw)
            assert match, (number, raw[:160])
            name, expression = match.groups()
            assert name not in local
            local[name] = f'V{len(local)}'
            roots += int(raw.lstrip().startswith('ROOT '))
            operation = OPCODE.search(expression)
            assert operation, (number, expression[:160])
            opcodes[operation.group(1)] += 1
            if operation.group(1) == 'constant' and '...' in expression:
                elisions.append(number)
            line = METADATA.sub('', raw.strip())
            assert 'metadata=' not in line, (number, 'Unsupported metadata syntax')

            def replace(symbol, local=local, number=number, callees=callees):
                key = symbol.group(1)
                assert key in local or key in functions, (number, 'Unknown HLO reference', key)
                if key in local:
                    return '%' + local[key]
                assert functions[key] is not None, (number, 'Forward/recursive computation', key)
                callees.add(key)
                return '%callee_' + functions[key]

            body.append(SYMBOL.sub(replace, line))
        else:
            assert not raw.strip() or raw.startswith('HloModule '), (number, raw[:160])
    assert record is None and records
    assert sum(row['entry'] for row in records) <= 1
    return records


def comparison(left, right):
    counts_a = collections.Counter(row['normalized_sha256'] for row in left)
    counts_b = collections.Counter(row['normalized_sha256'] for row in right)

    def unmatched(rows, counts):
        remaining, found = counts.copy(), []
        for row in rows:
            digest = row['normalized_sha256']
            if remaining[digest]:
                found.append(row)
                remaining[digest] -= 1
        return found

    return {'left_computations': len(left), 'right_computations': len(right),
        'matched_by_content_and_multiplicity': sum((counts_a & counts_b).values()),
        'left_unmatched': unmatched(left, counts_a - counts_b),
        'right_unmatched': unmatched(right, counts_b - counts_a)}


def analyze(raw, controls_run, output):
    controls_path = raw / f'run-{controls_run:05d}' / 'dz5-optimized-controls-readback.json'
    control_run_path = controls_path.parent / 'run.json'
    manifest = json.loads(control_run_path.read_text())
    assert manifest['state'] == 'passed' and manifest['device'] == 'CPU'
    controls = json.loads(controls_path.read_text())
    parsed, inputs = {}, {}
    for arm, row in controls['arms'].items():
        assert row['exact_all_callbacks_and_records']
        path = raw / row['run'] / 'optimized-context.hlo.txt'
        digest = sha(path)
        assert digest == row['digests'][path.name]
        with path.open() as stream:
            parsed[arm] = computations(stream)
        assert sum(record['entry'] for record in parsed[arm]) == 1
        inputs[arm] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
            'computations': len(parsed[arm]), 'instructions': sum(r['instruction_count'] for r in parsed[arm]),
            'opaque_constants': sum(len(r['opaque_constant_lines']) for r in parsed[arm]),
            'entry': next(r for r in parsed[arm] if r['entry'])}
        (output / f'{arm}-computations.json').write_text(json.dumps(parsed[arm], indent=2) + '\n')
    assert set(parsed) == {'original', 'candidate', 'replay_int32'}
    pairs = {f'{a}__{b}': comparison(parsed[a], parsed[b]) for a, b in (
        ('original', 'candidate'), ('original', 'replay_int32'), ('candidate', 'replay_int32'))}
    result = {'schema': 'filter_locator_optimized_hlo_structure.v1', 'inputs': inputs,
        'controls_path': str(controls_path), 'controls_sha256': sha(controls_path),
        'controls_manifest_sha256': sha(control_run_path), 'comparisons': pairs,
        'nonclaims': ['Exported optimized HLO may compile separately from the measured execution.',
            'Names/source metadata removed; printed layouts/constants/operand relations/attributes/instruction order/callee structure retained.',
            'TensorFlow text may elide constant payloads. Opaque sites are counted; matching printed structure never certifies literal or numerical identity.',
            'Content matching is structural evidence, not numerical equivalence or identification of a causal compiler pass.']}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return {name: {key: value if isinstance(value, int) else len(value)
        for key, value in row.items()} for name, row in pairs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--controls-run', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output_root / Path(__file__).name).write_bytes(source)
    started = time.monotonic()
    record = {'device': 'CPU', 'state': 'running', 'timeout_seconds': 180,
        'command': [sys.executable, *sys.argv], 'reader_sha256': hashlib.sha256(source).hexdigest()}
    destination = args.output_root / 'run.json'
    destination.write_text(json.dumps(record, indent=2) + '\n')
    try:
        summary = analyze(args.raw_root, args.controls_run, args.output_root)
        record['state'] = 'passed'
        print(json.dumps(summary))
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        destination.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
