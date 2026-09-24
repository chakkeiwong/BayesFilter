"""Independent synthetic diagnostic for per-candidate A11 report semantics."""
import importlib.util
from pathlib import Path

path = Path(__file__).with_name('inspect_result.py')
spec = importlib.util.spec_from_file_location('a11_reporting', path)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
healthy = dict(completed=12, reference_valid_sequences=12, failures={},
               log_evidence_failures=[], cdf_failures=0, consumer_invalid_steps=0,
               particles=dict(finite=True, cdf_invalid_steps=0))
summary = dict(sequences=12, reference_valid=12,
               arms={name: dict(healthy) for name in ('baseline', 'nominee', 'stronger-defense')})
inference = dict(promotion_veto=True,
    heuristic_losses=[dict(candidate='baseline', regime='large', heuristic='transition')],
    contrasts={'nominee': dict(simultaneous_interval=[-.3, -.1]),
               'stronger-defense': dict(simultaneous_interval=[-.2, .2])})
decisions = reporter.arm_decisions(summary, inference)
assert decisions['baseline']['promotion_veto']
assert not decisions['nominee']['promotion_veto']
assert decisions['nominee']['eligible_under_a11']
assert decisions['nominee']['primary_criterion'] == 'lower_mse'
assert not decisions['stronger-defense']['eligible_under_a11']
assert decisions['stronger-defense']['primary_criterion'] == 'non_harm_not_established'
summary['arms']['nominee']['failures'] = {'3': 'invalid density'}
decisions = reporter.arm_decisions(summary, inference)
assert decisions['nominee']['promotion_veto']
assert not decisions['nominee']['eligible_under_a11']
print('PASS: own failures veto their arm; comparator failures do not leak to another arm; uncertainty governs eligibility')
