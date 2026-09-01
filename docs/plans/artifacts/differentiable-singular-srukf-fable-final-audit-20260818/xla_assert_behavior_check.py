"""Audit check: do fail-closed asserts survive jit_compile=True (XLA)?"""
import numpy as np
import tensorflow as tf
from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower

bad = tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0e-14, 0.0, 0.0]]], tf.float64)  # relative pivot ~1e-14 < 1e-12

print("eager:")
try:
    batched_stack_qr_lower(bad)
    print("  NO ERROR (assert dropped)")
except Exception as exc:
    print("  raised:", type(exc).__name__, str(exc).splitlines()[0][:100])

print("jit_compile=True:")
fn = tf.function(lambda s: batched_stack_qr_lower(s)[0], jit_compile=True)
try:
    out = fn(bad).numpy()
    print("  NO ERROR (assert dropped under XLA); factor finite:", np.isfinite(out).all())
except Exception as exc:
    print("  raised:", type(exc).__name__, str(exc).splitlines()[0][:100])

print("graph (jit_compile=False):")
fn2 = tf.function(lambda s: batched_stack_qr_lower(s)[0], jit_compile=False)
try:
    out = fn2(bad).numpy()
    print("  NO ERROR; factor finite:", np.isfinite(out).all())
except Exception as exc:
    print("  raised:", type(exc).__name__, str(exc).splitlines()[0][:100])

# NaN input under XLA
nan_stack = tf.constant(np.nan, shape=[1, 2, 4], dtype=tf.float64)
print("NaN input, jit_compile=True:")
try:
    out = fn(nan_stack).numpy()
    print("  NO ERROR (assert dropped under XLA); output:", out.ravel()[:3])
except Exception as exc:
    print("  raised:", type(exc).__name__, str(exc).splitlines()[0][:100])
