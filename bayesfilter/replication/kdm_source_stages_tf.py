"""Source bearings initialization, weight pretraining and full-stage consumers.

Source: Younis b0e2fd5; see controlled-smc-kdm-source-stages-20261007.md.
These GPU/XLA consumers share the neural/KDE kernels and Adam variables. Their
existence does not establish full trained-method replication or qualification.
"""
import tensorflow as tf
from bayesfilter.replication.kdm_sampling_tf import SourceKDMResampler


class IndependentNormalVonMises:
    """Independent source distributions; concentration is computed by the caller."""
    def __init__(self, batch, count, dtype, jit_compile=True):
        self.batch, self.count, self.dtype = batch, count, dtype
        self.sampler = SourceKDMResampler(batch, 1, count, dtype=dtype,
                                          jit_compile=jit_compile)

    def draw(self, centers, scales, concentration, seed):
        key = lambda i: tf.random.experimental.stateless_fold_in(seed, i)
        scales = tf.broadcast_to(tf.constant(scales, self.dtype), [self.batch, 3])
        kappa = tf.fill([self.batch], tf.constant(concentration, self.dtype))
        particles = centers[:, None, :]
        valid = self.sampler._valid(particles, tf.ones([self.batch, 1], self.dtype), scales)
        def uniform(i):
            return tf.random.stateless_uniform([3]+self.sampler.shape, key(i+1),
                                               dtype=tf.float64, alg="philox")
        angle, iterations, done = self.sampler._circular(
            particles, scales, uniform, valid, concentration=kappa)
        noise = tf.random.stateless_normal([self.count, self.batch, 1, 2], key(0),
                                           dtype=self.dtype, alg="philox")
        normal = particles[None, ..., :2] + scales[None, :, None, :2]*noise
        values = tf.transpose(tf.concat([normal, angle[..., None]], -1)[:, :, 0, :], [1, 0, 2])
        valid &= done & tf.reduce_all(tf.math.is_finite(values))
        return values, valid, iterations


class SourceInitialCloud:
    def __init__(self, model):
        self.model = model
        b, n, d = model.batch, model.particles, model.dtype
        self.drawer = IndependentNormalVonMises(b, n, d, model.jit_compile)
        truth = tf.TensorSpec([b, 3], d)
        options = dict(jit_compile=model.jit_compile, autograph=False)
        self.replay = tf.function(self._replay, input_signature=[truth, tf.TensorSpec([b,n,3],d)], **options)
        self.sample = tf.function(self._sample, input_signature=[truth, tf.TensorSpec([2],tf.int64)], **options)

    def _replay(self, truth, noise):
        m = self.model
        # Source adds a VM(0, 1/.03) draw to the true angle without wrapping.
        particles = truth[:, None, :] + noise
        weights = tf.ones([m.batch,m.particles], m.dtype)/m.c(m.particles)
        band = tf.fill([m.batch,3], m.c(.001))
        resampling_band = m.bandwidth(True) if m.decoupled else band
        valid = tf.reduce_all(tf.math.is_finite(particles))
        valid &= tf.reduce_all(tf.math.is_finite(resampling_band) & (resampling_band > 0))
        return dict(particles=particles, particle_weights=weights, bandwidths=band,
                    resampling_particle_weights=weights, resampling_bandwidths=resampling_band,
                    noise=noise, valid=valid)

    def _sample(self, truth, seed):
        noise, valid, iterations = self.drawer.draw(
            tf.zeros_like(truth), [.5,.5,.03], 1.0/.03, seed)
        result = self._replay(truth, noise)
        result["valid"] &= valid
        result["iterations"] = iterations
        return result


class SourceFullTraining:
    def __init__(self, model):
        self.model = model
        self.initial = SourceInitialCloud(model)
        b, t, d = model.batch, model.horizon, model.dtype
        signature = [tf.TensorSpec([b,t,3],d), tf.TensorSpec([b,t,1],d),
                     tf.TensorSpec([b,t],tf.bool), tf.TensorSpec([2],tf.int64)]
        options = dict(input_signature=signature, jit_compile=model.jit_compile, autograph=False)
        self.update = tf.function(self._update, **options)
        self.evaluate = tf.function(self._evaluate, **options)

    def _loss(self, states, observations, mask, seed):
        key = lambda i: tf.random.experimental.stateless_fold_in(seed, i)
        init = self.initial._sample(states[:,0,:], key(0))
        inputs = [init[k] for k in ("particles", "particle_weights", "bandwidths",
                                   "resampling_particle_weights", "resampling_bandwidths")]
        loss, valid, outputs = self.model._native_sequence(
            *inputs, observations, states, mask, key(1))
        expected = tf.broadcast_to(tf.range(self.model.horizon)[None,:] % 4 == 0,
                                   [self.model.batch,self.model.horizon])
        valid &= init["valid"] & tf.reduce_all(mask == expected)
        return loss, valid, outputs

    def _update(self, *inputs):
        return self.model._apply_update(self._loss, inputs)

    def _evaluate(self, *inputs):
        loss, valid, outputs = self._loss(*inputs)
        return dict(loss=loss, valid=valid, outputs=outputs)


class SourceWeightPretraining:
    def __init__(self, model, sequence_batch):
        self.sequence_batch = int(sequence_batch)
        self.times = tuple(range(0, model.horizon, 4))
        self.model = model.for_batch(self.sequence_batch*len(self.times), particles=50)
        prefixes = ("weight_model.", "observation_encoder.", "particle_encoder_for_weights_model.")
        expected = {name for name in model.parameters if name.startswith(prefixes)}
        if set(model.names) != expected:
            raise ValueError("Source weight-pretraining optimizer coverage mismatch")
        self.center_drawer = IndependentNormalVonMises(self.model.batch, 1, model.dtype, model.jit_compile)
        self.cloud_drawer = IndependentNormalVonMises(self.model.batch, 50, model.dtype, model.jit_compile)
        b,t,d = self.sequence_batch, model.horizon, model.dtype
        signature = [tf.TensorSpec([b,t,3],d), tf.TensorSpec([b,t,1],d), tf.TensorSpec([b,t],tf.bool)]
        replay_signature = signature + [tf.TensorSpec([self.model.batch,50,3],d)]
        native_signature = signature + [tf.TensorSpec([2],tf.int64)]
        options = dict(jit_compile=model.jit_compile, autograph=False)
        self.update = tf.function(self._update, input_signature=replay_signature, **options)
        self.evaluate = tf.function(self._evaluate, input_signature=replay_signature, **options)
        self.native_update = tf.function(self._native_update, input_signature=native_signature, **options)
        self.native_evaluate = tf.function(self._native_evaluate, input_signature=native_signature, **options)

    def _select(self, values):
        # Author vstack selects time first, then the sequence batch.
        selected = tf.transpose(tf.gather(values, self.times, axis=1), [1,0,2])
        return tf.reshape(selected, [self.model.batch, values.shape[-1]])

    def _loss(self, states, observations, mask, cloud):
        m = self.model
        truth, observation = self._select(states), self._select(observations)
        obs = tf.concat([tf.sin(observation),tf.cos(observation)],-1)
        internal = tf.concat([cloud[...,:2],tf.sin(cloud[...,2:]),tf.cos(cloud[...,2:])],-1)
        obs_features = m.mlp(obs, "observation_encoder.encoder")
        obs_features = tf.broadcast_to(obs_features[:,None,:],[m.batch,50,8])
        features = m.mlp(internal, "particle_encoder_for_weights_model.encoder")
        logits = m.mlp(tf.concat([obs_features,features],-1),"weight_model.encoder")[...,0]
        raw = tf.sigmoid(logits)*m.c(1.0-.00001)+m.c(.00001)
        weights = raw/tf.maximum(tf.reduce_sum(tf.abs(raw),-1,keepdims=True),m.c(1e-8))
        band = tf.broadcast_to(m.c([1.,1.,.1]), [m.batch,3])
        posterior = m.kde._posterior(cloud,weights,band,truth)
        expected = tf.broadcast_to(tf.range(m.horizon)[None,:] % 4 == 0,[self.sequence_batch,m.horizon])
        valid = posterior["valid"] & tf.reduce_all(mask == expected)
        valid &= tf.reduce_all(tf.math.is_finite(cloud))
        return tf.reduce_mean(posterior["nll"]), valid, dict(
            particles=cloud, particle_weights=weights, bandwidths=band,
            selected_states=truth, selected_observations=observation,
            observation_features=obs_features, particle_features=features,
            logits=logits, nll=posterior["nll"])

    def _native_loss(self, states, observations, mask, seed):
        key = lambda i: tf.random.experimental.stateless_fold_in(seed,i)
        truth = self._select(states)
        center, center_valid, ci = self.center_drawer.draw(
            truth,[2.,2.,.1],1./(.1+1e-8),key(0))
        cloud, cloud_valid, pi = self.cloud_drawer.draw(
            center[:,0,:],[3.,3.,.1],1./(.1+1e-8),key(1))
        loss, valid, outputs = self._loss(states,observations,mask,cloud)
        outputs.update(cloud_centers=center[:,0,:], sampling_iterations=tf.stack([ci,pi]))
        return loss, valid & center_valid & cloud_valid, outputs

    def _update(self, *inputs):
        return self.model._apply_update(self._loss, inputs, gradient_clip=None, weight_decay=.001)

    def _native_update(self, *inputs):
        return self.model._apply_update(self._native_loss, inputs, gradient_clip=None, weight_decay=.001)

    def _evaluate(self, *inputs):
        loss, valid, outputs = self._loss(*inputs)
        return dict(loss=loss,valid=valid,outputs=outputs)

    def _native_evaluate(self, *inputs):
        loss, valid, outputs = self._native_loss(*inputs)
        return dict(loss=loss,valid=valid,outputs=outputs)
