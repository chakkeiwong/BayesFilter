"""CPU debug: isolate native-call accounting in cProfile."""
import cProfile,pstats
import tensorflow as tf
@tf.function(input_signature=[tf.TensorSpec([],tf.float64)],autograph=False)
def graph(x): return x*x

def outer():
    for i in range(3): graph(tf.constant(float(i),tf.float64))
for builtins in [True,False,True,False]:
    p=cProfile.Profile();p.enable(builtins=builtins);outer();p.disable()
    s=pstats.Stats(p)
    print(builtins, [(k[2],v[:4]) for k,v in s.stats.items() if k[2]=="outer"],flush=True)
