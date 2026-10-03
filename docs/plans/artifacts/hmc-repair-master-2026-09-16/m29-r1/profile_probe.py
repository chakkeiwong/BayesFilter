"""Debugging-only profile event check; CPU with GPU intentionally hidden."""
import cProfile, pstats, sys
import tensorflow as tf
print(sys.version,flush=True)
@tf.function(input_signature=[tf.TensorSpec([],tf.float64)])
def f(x): return x*x

def outer():
    for i in range(3):
        f(tf.constant(float(i),tf.float64))
    return 42
p=cProfile.Profile();p.enable();outer();p.disable()
for row in p.getstats():
    code=row.code
    if hasattr(code,"co_name") and code.co_name in {"outer","f"}: print(code,row.callcount,row.totaltime,flush=True)
pstats.Stats(p).sort_stats("cumulative").print_stats(5)
