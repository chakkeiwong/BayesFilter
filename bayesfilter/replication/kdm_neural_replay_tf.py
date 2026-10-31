"""Source-matched bearings neural KDM replay consumer in TensorFlow.

This implements the pinned source architecture, sparse full-sequence NLL,
truncated BPTT and Adam update for a fixed source checkpoint and draw tape.
Native resampling/noise share the replay numerical path. Source-stage consumers
share these parameters and kernels; complete training qualification is separate.
See controlled-smc-kdm-neural-consumer-20261006.md for scope/source anchors.
"""
import copy
import math
import tensorflow as tf
from bayesfilter.replication.kdm_sampling_tf import SourceKDMResampler
from bayesfilter.replication.kdm_normal_von_mises_tf import NormalVonMisesKDM


class BearingsNeuralReplay:
    def __init__(self, fixture, *, batch_size=None, particles=25, horizon=17,
                 jit_compile=True, capture_sampling_diagnostics=False):
        self.capture_sampling_diagnostics = bool(capture_sampling_diagnostics)
        self.dtype = tf.as_dtype(fixture["dtype"])
        batch_size = int(fixture.get("batch_size", 2) if batch_size is None else batch_size)
        self.batch, self.particles, self.horizon = batch_size, particles, horizon
        self.decoupled = fixture["variant"] == "am_dpf"
        if fixture["variant"] not in ("mdpf", "am_dpf") or horizon < 2:
            raise ValueError("Unsupported source replay scope")
        self.names = list(fixture["trainable_names"])
        self.parameters = {name: tf.Variable(value, dtype=self.dtype,
                           trainable=name in self.names, name=name.replace(".", "_"))
                           for name, value in fixture["parameters"].items()}
        self.trainable = [self.parameters[name] for name in self.names]
        self.variant = fixture["variant"]
        self.rates = [tf.Variable(tf.constant(fixture["learning_rates"][name], tf.float64), trainable=False)
                      for name in self.names]
        if set(self.names) != set(fixture["learning_rates"]):
            raise ValueError("Optimizer parameter coverage mismatch")
        self.moments1 = [tf.Variable(tf.zeros_like(p), trainable=False) for p in self.trainable]
        self.moments2 = [tf.Variable(tf.zeros_like(p), trainable=False) for p in self.trainable]
        self.step = tf.Variable(self.c(0), trainable=False)
        self.constants = {k: self.c(v) for k,v in fixture["constants"].items() if v is not None}
        self.used_parameter_names = set()
        self.jit_compile = bool(jit_compile)
        self._bind_graphs()

    def for_batch(self, batch_size, *, particles=None):
        """Specialize static shapes while sharing every parameter/Adam variable."""
        if int(batch_size) <= 0 or (particles is not None and int(particles) <= 0):
            raise ValueError("Positive batch/particle dimensions required")
        view = copy.copy(self)
        view.batch = int(batch_size)
        view.particles = self.particles if particles is None else int(particles)
        view._bind_graphs()
        return view

    def _bind_graphs(self):
        batch_size, particles, horizon = self.batch, self.particles, self.horizon
        jit_compile = self.jit_compile
        self.kde = NormalVonMisesKDM(batch_size=batch_size, particles=particles,
                                    circular_dimensions=[False, False, True],
                                    dtype=self.dtype, jit_compile=jit_compile)
        b,n,t = batch_size, particles, horizon
        tensor = lambda shape: tf.TensorSpec(shape, self.dtype)
        signature = [tensor([b,n,3]), tensor([b,n]), tensor([b,3]),
                     tensor([b,n]), tensor([b,3]), tensor([b,t,1]),
                     tensor([b,t,3]), tf.TensorSpec([b,t], tf.bool),
                     tensor([t-2,b,n,3]), tensor([t-1,b,n,4])]
        self.update = tf.function(self._update, input_signature=signature,
                                  jit_compile=jit_compile, autograph=False)
        self.evaluate = tf.function(self._evaluate, input_signature=signature,
                                    jit_compile=jit_compile, autograph=False)
        self.sampler = SourceKDMResampler(b, n, n, dtype=self.dtype,
                                          jit_compile=jit_compile)
        native_signature = signature[:8] + [tf.TensorSpec([2], tf.int64)]
        self.native_update = tf.function(self._native_update, input_signature=native_signature,
                                         jit_compile=jit_compile, autograph=False)
        self.native_evaluate = tf.function(self._native_evaluate, input_signature=native_signature,
                                           jit_compile=jit_compile, autograph=False)

    def set_learning_rates(self, rates):
        if set(rates) != set(self.names):
            raise ValueError("Learning-rate parameter coverage mismatch")
        values = [float(rates[name]) for name in self.names]
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("Invalid learning rate")
        for variable, value in zip(self.rates, values):
            variable.assign(tf.constant(value, variable.dtype))

    def state_dict(self):
        # Host serialization only; no NumPy numerical computation.
        return dict(schema="kdm_neural_training_state_v2", variant=self.variant,
                    adam_scalar_policy="pytorch_noncapturable_binary64_v1",
                    dtype=self.dtype.name, trainable_names=self.names,
                    constants={k:v.numpy().tolist() for k,v in self.constants.items()},
                    parameters={k:v.numpy().tolist() for k,v in self.parameters.items()},
                    rates={k:float(v.numpy()) for k,v in zip(self.names,self.rates)},
                    first_moments={k:v.numpy().tolist() for k,v in zip(self.names,self.moments1)},
                    second_moments={k:v.numpy().tolist() for k,v in zip(self.names,self.moments2)},
                    step=float(self.step.numpy()))

    def load_state_dict(self, state):
        if (state["schema"] != "kdm_neural_training_state_v2"
                or state.get("adam_scalar_policy") != "pytorch_noncapturable_binary64_v1"
                or state["variant"] != self.variant or state["dtype"] != self.dtype.name
                or state["trainable_names"] != self.names
                or state["constants"] != {k:v.numpy().tolist() for k,v in self.constants.items()}):
            raise ValueError("Incompatible neural checkpoint")
        groups = [(self.parameters, state["parameters"]),
                  (dict(zip(self.names,self.moments1)),state["first_moments"]),
                  (dict(zip(self.names,self.moments2)),state["second_moments"]),
                  (dict(zip(self.names,self.rates)),state["rates"])]
        assignments = []
        for variables, values in groups:
            if set(variables) != set(values):
                raise ValueError("Incomplete neural checkpoint")
            for name, variable in variables.items():
                value = tf.convert_to_tensor(values[name], variable.dtype)
                if value.shape != variable.shape or not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
                    raise ValueError("Invalid neural checkpoint tensor")
                assignments.append((variable,value))
        if not math.isfinite(state["step"]) or state["step"] < 0 or state["step"] != int(state["step"]):
            raise ValueError("Invalid optimizer step")
        if any(float(x) < 0 for x in state["rates"].values()):
            raise ValueError("Invalid saved learning rate")
        if any(not bool(tf.reduce_all(value >= 0).numpy()) for _,value in
               assignments[len(self.parameters)+len(self.names):len(self.parameters)+2*len(self.names)]):
            raise ValueError("Invalid Adam second moment")
        # Validate every field before mutating any state.
        for variable, value in assignments:
            variable.assign(value)
        self.step.assign(self.c(state["step"]))

    def c(self, value):
        return tf.constant(value, self.dtype)

    def parameter(self, name):
        self.used_parameter_names.add(name)
        return self.parameters[name]

    def mlp(self, x, prefix):
        # Static architecture layers, never a Python loop over time or particles.
        for layer in (0,2,4):
            if layer:
                x = tf.nn.relu(x)
            x = tf.linalg.matmul(x, self.parameter(prefix+f".{layer}.weight"), transpose_b=True)
            x += self.parameter(prefix+f".{layer}.bias")
        return x

    def bandwidth(self, resampling=False):
        prefix = "resampling_weighted_bandwidth_predictor" if resampling else "weighted_bandwidth_predictor"
        key = "resampling_min_bandwidths" if resampling else "min_bandwidths"
        band = tf.exp(self.parameter(prefix+".log_bandwidths")) + self.constants[key]
        return tf.broadcast_to(band, [self.batch,3])

    def transition(self, previous, weights, band, resampling_weights, resampling_band,
                   observation, sample, noise, first):
        def draw_score():
            result = self.kde._resampling(previous, resampling_weights, resampling_band, sample)
            return tf.stop_gradient(sample), result["iwsg"], result["valid"]
        # The source's first transition has a different, fixed operation.
        # Select it at trace time: a runtime cond around the differentiated
        # I0 loop produces unsupported variant FakeParam nodes in XLA.
        cloud, iwsg, valid = ((previous, resampling_weights, tf.constant(True))
                             if first else draw_score())
        cloud = tf.concat([tf.clip_by_value(cloud[...,:2], self.c(-10000), self.c(10000)),
                           cloud[...,2:]], -1)
        internal = tf.concat([cloud[...,:2], tf.sin(cloud[...,2:]), tf.cos(cloud[...,2:])], -1)
        encoded = self.mlp(internal[...,2:], "particle_encoder_for_particles_model.encoder")
        proposal = self.mlp(tf.concat([encoded,noise], -1), "proposal_model.proposal_model")
        internal += (tf.sigmoid(proposal)*self.c(2)-self.c(1))*self.constants["residual_scale"]
        proposal_before_normalization = internal
        magnitude = tf.sqrt(tf.square(internal[...,2])+tf.square(internal[...,3]))
        valid &= tf.reduce_all(tf.math.is_finite(magnitude) & (magnitude > 0))
        internal = tf.concat([tf.clip_by_value(internal[...,:2], self.c(-10000), self.c(10000)),
                              internal[...,2:]/magnitude[...,None]], -1)
        locations = tf.concat([internal[...,:2], tf.atan2(internal[...,2:3],internal[...,3:4])], -1)
        obs_features = self.mlp(observation, "observation_encoder.encoder")
        obs_features = tf.broadcast_to(obs_features[:,None,:], [self.batch,self.particles,8])
        particle_features = self.mlp(internal, "particle_encoder_for_weights_model.encoder")
        inputs = tf.concat([obs_features,particle_features], -1)
        def weigh(prefix):
            logits = self.mlp(inputs,prefix)[...,0]
            raw = (tf.sigmoid(logits)*self.c(1.0-0.00001)+self.c(0.00001))*iwsg
            return raw/tf.maximum(tf.reduce_sum(tf.abs(raw),-1,keepdims=True),self.c(1e-8))
        new_weights = weigh("weight_model.encoder")
        new_band = self.bandwidth()
        new_resampling_weights = weigh("resampling_weight_model.encoder") if self.decoupled else new_weights
        new_resampling_band = self.bandwidth(True) if self.decoupled else new_band
        valid &= tf.reduce_all(tf.math.is_finite(locations))
        valid &= self.kde._weight_valid(new_weights) & self.kde._weight_valid(new_resampling_weights)
        return (locations,new_weights,new_band,new_resampling_weights,
                new_resampling_band,valid,proposal_before_normalization)

    def _sequence(self, locations, weights, band, resampling_weights, resampling_band,
                  observations, states, mask, samples, noise, native_seed=None):
        observations = tf.concat([tf.sin(observations),tf.cos(observations)], -1)
        output_shapes = ([self.batch,self.particles,3], [self.batch,self.particles],
                         [self.batch,3], [self.batch,self.particles], [self.batch,3],
                         [self.batch,self.particles,4], [self.batch,self.particles,3],
                         [self.batch,self.particles,4], [])
        if self.capture_sampling_diagnostics:
            # Preserve choices from the actual sampler call; recomputing a CDF
            # outside this XLA kernel can round a boundary differently.
            output_shapes += ([self.batch,self.particles],) * 3
        outputs = tuple(tf.TensorArray(self.dtype,size=self.horizon-1,
                        element_shape=shape) for shape in output_shapes)
        # Source initial state has do_resample=False; dummy draw is unconsumed.
        if native_seed is None:
            samples = tf.concat([locations[None,...],samples],0)
        def body(i,p,w,b,rw,rb,loss_sum,loss_count,valid,*arrays,first=False):
            detach = (i != 1) & (tf.math.floormod(i-1,4) == 0)
            p,w,b,rw,rb = tf.cond(detach,
                lambda: tuple(tf.stop_gradient(x) for x in (p,w,b,rw,rb)),
                lambda: (p,w,b,rw,rb))
            draw_valid, iterations = tf.constant(True), tf.constant(0)
            if self.capture_sampling_diagnostics:
                sample_diagnostics = (tf.fill([self.batch,self.particles],self.c(-1)),
                                      tf.zeros([self.batch,self.particles],self.dtype),
                                      tf.zeros([self.batch,self.particles],self.dtype))
            if native_seed is None:
                draw, innovation = samples[i-1], noise[i-1]
            else:
                key = lambda tag: tf.random.experimental.stateless_fold_in(native_seed, tag)
                innovation = tf.random.stateless_normal(
                    [self.batch,self.particles,4],key(1000+i),dtype=self.dtype,alg="philox")
                if first:
                    draw = p
                else:
                    sampled = self.sampler._sample(
                        tf.stop_gradient(p), tf.stop_gradient(rw), tf.stop_gradient(rb), key(i))
                    draw, draw_valid, iterations = sampled["samples"], sampled["valid"], sampled["iterations"]
                    if self.capture_sampling_diagnostics:
                        sample_diagnostics = (
                            tf.cast(sampled["ancestors"],self.dtype),
                            sampled.get("categorical_cdf",sample_diagnostics[1]),
                            sampled.get("categorical_uniform",sample_diagnostics[2]))
            p,w,b,rw,rb,step_valid,raw_proposal = self.transition(
                p,w,b,rw,rb,observations[:,i,:],draw,innovation,first)
            step_valid &= draw_valid
            posterior = self.kde._posterior(p,w,b,states[:,i,:])
            selected = tf.cast(mask[:,i],self.dtype)
            loss_sum += tf.reduce_sum(posterior["nll"]*selected)
            # Author averages masked batch vectors over selected time indices,
            # not over the number of individually selected states.
            loss_count += tf.where(tf.reduce_any(mask[:,i]), self.batch, 0)
            values = (p,w,b,rw,rb,raw_proposal,draw,innovation,tf.cast(iterations,self.dtype))
            if self.capture_sampling_diagnostics:
                values += sample_diagnostics
            arrays = tuple(a.write(i-1,x) for a,x in zip(arrays,values))
            return (i+1,p,w,b,rw,rb,loss_sum,loss_count,
                    valid & step_valid & posterior["valid"],*arrays)
        # Peel the first source transition; retain one traced body for all
        # subsequent times and the identical source BPTT detach boundaries.
        initial = body(tf.constant(1),locations,weights,band,
                       resampling_weights,resampling_band,self.c(0),
                       tf.constant(0),tf.constant(True),*outputs,first=True)
        result = tf.while_loop(lambda i,*rest:i<self.horizon, body, initial,
                               maximum_iterations=self.horizon-2)
        loss = result[6]/tf.cast(result[7],self.dtype)
        return loss,result[8] & (result[7]>0),tuple(a.stack() for a in result[9:])

    def _native_sequence(self, *inputs):
        return self._sequence(*inputs[:8], None, None, native_seed=inputs[8])

    def _evaluate(self, *inputs):
        loss, valid, outputs = self._sequence(*inputs)
        return dict(loss=loss, valid=valid, outputs=outputs)

    def _native_evaluate(self, *inputs):
        loss, valid, outputs = self._native_sequence(*inputs)
        return dict(loss=loss, valid=valid, outputs=outputs)

    def _update(self, *inputs):
        return self._apply_update(self._sequence, inputs)

    def _native_update(self, *inputs):
        return self._apply_update(self._native_sequence, inputs)

    def _apply_update(self, sequence, inputs, *, gradient_clip=100.0, weight_decay=0.0):
        with tf.GradientTape() as tape:
            loss,valid,outputs = sequence(*inputs)
        gradients = tape.gradient(loss,self.trainable)
        if any(g is None for g in gradients):
            raise ValueError("Missing trainable gradient")
        norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in gradients]))
        coefficient = (self.c(1) if gradient_clip is None else
                       tf.minimum(self.c(1),self.c(gradient_clip)/(norm+self.c(1e-6))))
        clipped = [g*coefficient for g in gradients]
        optimizer_gradients = [g + self.c(weight_decay)*p
                               for g,p in zip(clipped,self.trainable)]
        t = self.step + self.c(1)
        # Source CPU Adam (torch.optim.adam, noncapturable branch) computes
        # beta powers and lr / bias_correction in Python binary64, then casts
        # scalar operands at tensor operations. Parameter-dtype powers alter
        # float32 updates before any network-gradient discrepancy is involved.
        scalar_step = tf.cast(t, tf.float64)
        correction1 = tf.constant(1., tf.float64)-tf.pow(tf.constant(.9, tf.float64),scalar_step)
        correction2 = tf.cast(tf.sqrt(tf.constant(1., tf.float64)-
                                     tf.pow(tf.constant(.999, tf.float64),scalar_step)),self.dtype)
        proposed = []
        valid &= tf.math.is_finite(loss) & tf.math.is_finite(norm)
        for p,g,m,v,lr in zip(self.trainable,optimizer_gradients,self.moments1,self.moments2,self.rates):
            # Match source Adam epsilon after bias-correcting sqrt(second moment).
            new_m = m+self.c(1-.9)*(g-m)
            new_v = self.c(.999)*v+self.c(1-.999)*tf.square(g)
            denominator = tf.sqrt(new_v)/correction2+self.c(1e-8)
            step_size = tf.cast(lr/correction1, self.dtype)
            new_p = p-step_size*(new_m/denominator)
            proposed.append((new_p,new_m,new_v))
            valid &= tf.reduce_all(tf.math.is_finite(g)) & tf.reduce_all(tf.math.is_finite(new_p))
            valid &= tf.reduce_all(tf.math.is_finite(new_m)) & tf.reduce_all(tf.math.is_finite(new_v))
        def commit():
            for p,m,v,(new_p,new_m,new_v) in zip(self.trainable,self.moments1,self.moments2,proposed):
                p.assign(new_p)
                m.assign(new_m)
                v.assign(new_v)
            self.step.assign(t)
            return tf.constant(True)
        committed = tf.cond(valid, commit, lambda: tf.constant(False))
        return dict(loss=loss,valid=valid,committed=committed,outputs=outputs,gradient_norm=norm,
                    raw_gradients=gradients,clipped_gradients=clipped,
                    optimizer_gradients=optimizer_gradients,
                    parameters_after=[p.read_value() for p in self.trainable],
                    first_moments=[m.read_value() for m in self.moments1],
                    second_moments=[v.read_value() for v in self.moments2],
                    step=self.step.read_value())
