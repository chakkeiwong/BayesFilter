"""Tiny CPU-only engineering checks, not sampler-validity evidence."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","1")
from concurrent.futures import ThreadPoolExecutor
import unittest
import tensorflow as tf
from validate import Native
from wrapper import FitConfig,compiled_fit

class BridgeContract(unittest.TestCase):
    def test_invalid_static_configuration(self):
        for change in ({"kind":9},{"dimension":0},{"chains":0},{"depth":16},
                       {"warmup":0},{"draws":10000001},{"kind":2,"dimension":3},
                       {"kind":3,"dimension":1}):
            values=dict(engine="stan",kind=0,dimension=4,chains=2,warmup=5,draws=5,depth=5)
            values.update(change)
            with self.assertRaises(ValueError):FitConfig(**values)

    def test_native_reentrancy(self):
        native=Native();starts=[.2,-.3];inverse=[1.,0.,0.,1.]
        configs=[FitConfig(e,0,2,1,10,10,5) for e in ("stan","walnutpie")]
        def run(pair):
            engine,seed=pair;out=native.fit(configs[engine],[.3,.1,.8,.5],starts,inverse,[seed])
            return tuple(bytes(x) for x in out)
        jobs=[(0,101),(1,202),(0,303),(1,404)]
        expected=[run(j) for j in jobs]
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(list(executor.map(run,jobs)),expected)

    def test_compiled_consumer_and_dynamic_seeds(self):
        native=Native();d=2;c=1;w=s=10
        for engine in ("stan","walnutpie"):
            cfg=FitConfig(engine,0,d,c,w,s,5);fit=compiled_fit(cfg)
            controls=tf.constant([.3,.1,.8,.5],tf.float64);starts=tf.constant([[.2,-.3]],tf.float64);inv=tf.eye(2,dtype=tf.float64)
            @tf.function(input_signature=[tf.TensorSpec([1],tf.int64)],jit_compile=True,autograph=False)
            def consumer(seed):
                rows,meta=fit(controls,starts,inv,seed)
                return rows[:,w:,7:]+1.,meta[:,:d+1]
            for seed in (101,202):
                rows,meta=native.fit(cfg,[.3,.1,.8,.5],[.2,-.3],[1.,0.,0.,1.],[seed])
                actual,m=consumer(tf.constant([seed],tf.int64))
                expected=[rows[i*(d+7)+7+j]+1. for i in range(w,w+s) for j in range(d)]
                self.assertEqual(tf.reshape(actual,[-1]).numpy().tolist(),expected)
                self.assertEqual(tf.reshape(m,[-1]).numpy().tolist(),list(meta)[:d+1])
            self.assertEqual(consumer.experimental_get_tracing_count(),1)

    def test_native_rejects_invalid_dynamic_inputs(self):
        native=Native();cfg=FitConfig("stan",0,2,1,5,5,5)
        for controls,starts,inverse,seeds in [
            ([.3,-1.,.8,.5],[0.,0.],[1.,0.,0.,1.],[1]),
            ([.3,.1,.8,.5],[float("nan"),0.],[1.,0.,0.,1.],[1]),
            ([.3,.1,.8,.5],[0.,0.],[1.,0.,0.,float("inf")],[1]),
            ([.3,.1,.8,.5],[0.,0.],[1.,0.,0.,1.],[2**32])]:
            with self.assertRaises(ValueError):native.fit(cfg,controls,starts,inverse,seeds)

if __name__=="__main__":unittest.main(verbosity=2)
