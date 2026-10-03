"""Tiny CPU debug comparison of profiler event collection around TensorFlow."""
import cProfile, profile, pstats, time
import tensorflow as tf
@tf.function(input_signature=[tf.TensorSpec([],tf.float64)],autograph=False)
def graph(x): return x*x

def inner(): return sum(range(100))
def pure(): return inner()
def eager(): return tf.constant(2.,tf.float64)*tf.constant(3.,tf.float64)
def compiled(): return graph(tf.constant(2.,tf.float64))
for call in [pure,eager,compiled]:
    for kind in ["cProfile","python_profile"]:
        p=cProfile.Profile() if kind=="cProfile" else profile.Profile()
        p.runcall(call)
        s=pstats.Stats(p)
        print(kind,call.__name__, [(k[2],v[:4]) for k,v in s.stats.items() if k[2] in {"pure","inner","eager","compiled","graph"}],flush=True)
