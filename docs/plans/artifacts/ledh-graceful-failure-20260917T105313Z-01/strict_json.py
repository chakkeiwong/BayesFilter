"""Diagnostic JSON encoding: preserve nonfinite values as tagged objects."""
import math

def strict_numbers(value):
    if isinstance(value, dict):
        return {k: strict_numbers(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_numbers(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite_float__': 'NaN' if math.isnan(value) else
                ('Infinity' if value > 0 else '-Infinity')}
    return value
