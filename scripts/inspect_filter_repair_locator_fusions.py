"""Diagnostic caller trace for the saved locator's two floating fusion variants.

This inspects printed IR, never executes or changes a numerical program. Large
literal payloads elsewhere in the module remain opaque; this is not equivalence
or compiler-pass attribution evidence.
"""

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

from scripts.analyze_filter_repair_locator_optimized_hlo import (
    DEFINITION,
    HEADER,
    METADATA,
    OPCODE,
    SYMBOL,
    sha,
)


def instruction(line):
    match = DEFINITION.match(line)
    assert match, line[:120]
    name, expression = match.groups()
    expression = METADATA.sub('', expression)
    assert 'metadata=' not in expression
    opcode = OPCODE.search(expression)
    assert opcode, expression[:120]
    return {'name': name, 'expression': expression, 'opcode': opcode.group(1),
        'references': SYMBOL.findall(expression)}


def tuple_index(node):
    assert node['opcode'] == 'get-tuple-element'
    match = re.search(r', index=(\d+)$', node['expression'])
    assert match, node['expression'][:120]
    return int(match.group(1))


def invariant_inputs(body_lines, entry_lines, selected_fusion):
    """Trace fusion parameters 4/5 through the loop tuple to its initial values."""
    body = {node['name']: node for node in map(instruction, body_lines)}
    entry = {node['name']: node for node in map(instruction, entry_lines)}
    calls = [node for node in body.values() if node['opcode'] == 'fusion'
        and node['references'][-1] == selected_fusion]
    assert len(calls) == 1
    roots = [instruction(line) for line in body_lines if line.lstrip().startswith('ROOT ')]
    assert len(roots) == 1 and roots[0]['opcode'] == 'tuple'
    loop_parameter = next(node['name'] for node in body.values() if node['opcode'] == 'parameter')
    body_name = HEADER.match(body_lines.header).group(1)
    loops = [node for node in entry.values() if node['opcode'] == 'while'
        and re.search(r', body=%' + re.escape(body_name) + r'(?:,|$)', node['expression'])]
    assert len(loops) == 1
    initial = entry[loops[0]['references'][0]]
    assert initial['opcode'] == 'tuple'
    result = []
    for parameter in (4, 5):
        operand = body[calls[0]['references'][parameter]]
        index = tuple_index(operand)
        assert operand['references'] == [loop_parameter]
        returned = body[roots[0]['references'][index]]
        assert tuple_index(returned) == index and returned['references'] == [loop_parameter]
        literal = entry[initial['references'][index]]
        assert literal['opcode'] == 'constant' and '...' not in literal['expression']
        result.append({'fusion_parameter': parameter, 'tuple_index': index,
            'unchanged_by_loop_body': True, 'initial_literal': literal})
    return {'call': calls[0], 'loop': loops[0]['name'],
        'initial_tuple_size': len(initial['references']), 'invariants': result}


class Body(list):
    """Instruction lines with their exact printed computation header."""
    def __init__(self, header, lines):
        super().__init__(lines)
        self.header = header


def analyze(structure_root, output):
    path = structure_root / 'comparison.json'
    report = json.loads(path.read_text())
    assert sha(Path(report['controls_path'])) == report['controls_sha256']
    assert sha(Path(report['controls_path']).parent / 'run.json') == report['controls_manifest_sha256']
    pair = report['comparisons']['candidate__replay_int32']
    signatures = {row['normalized_sha256']: row['instruction_count']
        for side in ('left_unmatched', 'right_unmatched') for row in pair[side]
        if row['name'].startswith('fused_computation.')}
    assert sorted(signatures.values()) == [214, 217], signatures
    arms = {}
    for arm, source in report['inputs'].items():
        source_path = Path(source['path'])
        assert sha(source_path) == source['sha256']
        lines = source_path.read_text().splitlines()
        records_path = structure_root / f'{arm}-computations.json'
        records = json.loads(records_path.read_text())
        targets = {row['name']: row for row in records if row['normalized_sha256'] in signatures}
        selected = {row['name']: row for row in records if row['name'] in targets
            or set(row['callees']) & targets.keys() or row['entry']}
        counts = collections.Counter(signatures[row['normalized_sha256']] for row in targets.values())
        details = []
        for row in selected.values():
            excerpt = lines[row['line'] - 1:row['end_line']]
            (output / f'{arm}-{row["line"]}.hlo.txt').write_text('\n'.join(excerpt) + '\n')
            details.append({'name': row['name'], 'line': row['line'],
                'end_line': row['end_line'], 'instruction_count': row['instruction_count'],
                'normalized_sha256': row['normalized_sha256'],
                'selected_callees': sorted(set(row['callees']) & targets.keys()),
                'excerpt_sha256': sha(output / f'{arm}-{row["line"]}.hlo.txt')})
        arms[arm] = {'variant_counts': dict(counts), 'records_sha256': sha(records_path),
            'HLO_sha256': source['sha256'], 'computations': details}
        if arm == 'candidate':
            variant = {name for name, row in targets.items() if row['instruction_count'] == 217}
            callers = [row for row in records if set(row['callees']) & variant]
            assert len(callers) == 1 and len(variant) == 4
            row = callers[0]
            body = Body(lines[row['line'] - 1], lines[row['line']:row['end_line'] - 1])
            entry = next(row for row in records if row['entry'])
            entry_lines = lines[entry['line']:entry['end_line'] - 1]
            arms[arm]['parameter_invariant_traces'] = {
                name: invariant_inputs(body, entry_lines, name) for name in sorted(variant)}
    result = {'schema': 'filter_locator_fusion_inspection.v1',
        'structure_path': str(path), 'structure_sha256': sha(path), 'arms': arms,
        'nonclaims': ['Printed constants do not certify opaque literal payloads elsewhere.',
            'Independent export may compile separately from the measured call.',
            'Different fusion boundaries identify a mechanism to test, not a causal pass or runtime remedy.',
            'Historical r1 includes frozen derivative machinery; no current-source or pfor admission.']}
    (output / 'inspection.json').write_text(json.dumps(result, indent=2) + '\n')
    return {arm: row['variant_counts'] for arm, row in arms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structure-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    record = {'device': 'CPU', 'state': 'running', 'timeout_seconds': 120,
        'command': [sys.executable, '-m', 'scripts.inspect_filter_repair_locator_fusions', *sys.argv[1:]],
        'reader_sha256': sha(Path(__file__)),
        'parser_sha256': sha(Path(__file__).with_name('analyze_filter_repair_locator_optimized_hlo.py'))}
    destination = args.output_root / 'run.json'
    destination.write_text(json.dumps(record, indent=2) + '\n')
    try:
        print(json.dumps(analyze(args.structure_root, args.output_root)))
        record['state'] = 'passed'
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        destination.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
