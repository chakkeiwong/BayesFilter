"""Source KDM mixture sampling on TF GPU/XLA, with explicit rejection status.

Source anchors and scope: controlled-smc-kdm-native-sampling-20261006.md.
Callers must reject valid=False; the rejection cap never silently fills samples.
"""
import math
import tensorflow as tf


class SourceKDMResampler:
    def __init__(self, batch_size, particles, sample_count, dtype=tf.float32,
                 max_iterations=128, jit_compile=True):
        self.b, self.n, self.s = batch_size, particles, sample_count
        self.dtype = tf.as_dtype(dtype)
        self.limit = max_iterations
        if min(batch_size, particles, sample_count, max_iterations) <= 0:
            raise ValueError("Positive sampler dimensions and iteration cap required")
        self.shape = [sample_count, batch_size, particles]
        spec = lambda shape: tf.TensorSpec(shape, self.dtype)
        common = [spec([batch_size,particles,3]), spec([batch_size,particles]),
                  spec([batch_size,3])]
        self.sample = tf.function(self._sample, input_signature=common+
            [tf.TensorSpec([2],tf.int64)], autograph=False,jit_compile=jit_compile)
        self.replay = tf.function(self._replay, input_signature=common+[
            tf.TensorSpec([batch_size,sample_count],tf.int32),
            spec(self.shape+[2]),
            tf.TensorSpec([max_iterations,3]+self.shape,tf.float64)],
            autograph=False,jit_compile=jit_compile)

    def _valid(self,p,w,bw):
        return (tf.reduce_all(tf.math.is_finite(p)) &
                tf.reduce_all(tf.math.is_finite(w) & (w>=0)) &
                tf.reduce_all(tf.reduce_sum(w,-1)>0) &
                tf.reduce_all(tf.math.is_finite(bw) & (bw>0)))

    def _circular(self, p, bw, uniform, input_valid, concentration=None):
        # The source forms kappa in the model dtype, then promotes to double.
        if concentration is None:
            concentration = 1/(bw[:,-1]+tf.constant(1e-8,self.dtype))
        kappa=tf.cast(concentration,tf.float64)[None,:,None]
        loc=tf.cast(p[...,2],tf.float64)[None,:,:]
        tau=1+tf.sqrt(1+4*tf.square(kappa))
        rho=(tau-tf.sqrt(2*tau))/(2*kappa)
        r=tf.where(kappa<1e-5,1/kappa+kappa,(1+tf.square(rho))/(2*rho))
        def body(i,done,x):
            u=uniform(i)
            z=tf.cos(tf.constant(math.pi,tf.float64)*u[0])
            f=(1+r*z)/(r+z)
            c=kappa*(r-f)
            accept=(c*(2-c)-u[1]>0) | (tf.math.log(c/u[1])+1-c>=0)
            # Match source: later accepted proposals may replace earlier ones.
            x=tf.where(accept,tf.sign(u[2]-.5)*tf.acos(f),x)
            return i+1,done|accept,x
        iterations,done,x=tf.while_loop(
            lambda i,done,x: input_valid & ~tf.reduce_all(done) & (i<self.limit),
            body,(tf.constant(0),tf.zeros(self.shape,tf.bool),
                  tf.zeros(self.shape,tf.float64)),maximum_iterations=self.limit)
        angle=tf.math.floormod(x+tf.constant(math.pi,tf.float64)+loc,
                              tf.constant(2*math.pi,tf.float64))-tf.constant(math.pi,tf.float64)
        return tf.cast(angle,self.dtype),iterations,tf.reduce_all(done)

    def _assemble(self,p,w,bw,ancestors,normal,uniform):
        input_valid=self._valid(p,w,bw)
        angle,iterations,done=self._circular(p,bw,uniform,input_valid)
        components=tf.concat([normal,angle[...,None]],-1)
        samples=tf.gather(tf.transpose(components,[1,0,2,3]),ancestors,
                          axis=2,batch_dims=2)
        valid=input_valid & done & tf.reduce_all(tf.math.is_finite(samples))
        return dict(samples=tf.stop_gradient(samples),ancestors=ancestors,
                    iterations=iterations,accepted_all=done,valid=valid)

    def _sample(self,p,w,bw,seed):
        key=lambda i:tf.random.experimental.stateless_fold_in(seed,i)
        ancestors=tf.random.stateless_categorical(tf.math.log(w),self.s,key(0),
                                                  dtype=tf.int32)
        z=tf.random.stateless_normal(self.shape+[2],key(1),dtype=self.dtype,alg="philox")
        normal=p[None,:,:,:2]+bw[None,:,None,:2]*z
        def uniform(i):
            return tf.random.stateless_uniform([3]+self.shape,key(i+2),
                                               dtype=tf.float64,alg="philox")
        return self._assemble(p,w,bw,ancestors,normal,uniform)

    def _replay(self,p,w,bw,ancestors,normal,uniform_tape):
        return self._assemble(p,w,bw,ancestors,normal,lambda i:uniform_tape[i])
