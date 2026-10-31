"""Host control and exact checkpoint I/O for the frozen full80 comparison."""
import math

POLICY = dict(id='full80_walnutpie_author_fixed_identity_v1',
    initial_step=.1,target_macro_quality=.8,warmup_min=50,warmup_max=1000,
    warmup_stride=5,step_tolerance=.1,mass_tolerance=1.,chunk_size=5,
    retained_min=1400,retained_max=10000,assessment_stride=100,
    rhat_max=1.01,bulk_ess_min=400.,tail_ess_min=400.,mcse_sd_max=.05)

def adaptation_decision(tf, state, policy=POLICY, statistics_fn=None):
    if statistics_fn is None:
        from docs.benchmarks.walnutpie_gpu_tf.control import warmup_statistics
        statistics_fn=warmup_statistics
    tf.debugging.assert_equal(state.inverse_mass,tf.ones_like(state.inverse_mass))
    count=int(state.iteration)
    if count < 0 or count > policy['warmup_max']:
        raise ValueError('adaptation count outside source bounds')
    sd,md=statistics_fn(tf,state)
    tf.debugging.assert_all_finite(sd,'step agreement')
    tf.debugging.assert_all_finite(md,'mass agreement')
    ready=count>=policy['warmup_min'] and (count%policy['warmup_stride']==0 or count==policy['warmup_max'])
    agreement=ready and bool((sd<=policy['step_tolerance']) & (md<=policy['mass_tolerance']))
    capped=count==policy['warmup_max']
    return dict(freeze=agreement or capped,iterations=count,parameter_agreement=agreement,
        reason='parameter_agreement' if agreement else 'maximum' if capped else 'continue',
        maximum_relative_step_difference=float(sd),mass_difference=float(md),
        posterior_stationarity_established=False)

def checkpoint(tf, state, directory, prefix):
    from docs.benchmarks.full80_walnutpie_support import sha
    records=[]
    for i,value in enumerate(tf.nest.flatten(state)):
        p=directory/(prefix+'-'+str(i)+'.tensor')
        p.write_bytes(tf.io.serialize_tensor(value).numpy())
        records.append(dict(path=p.name,sha256=sha(p),shape=value.shape.as_list(),dtype=value.dtype.name))
    return records

def restore_checkpoint(tf, records, directory, spec):
    from docs.benchmarks.full80_walnutpie_support import sha
    specs=tf.nest.flatten(spec)
    if len(records)!=len(specs):raise ValueError('checkpoint length mismatch')
    values=[]
    for row,expected in zip(records,specs):
        p=(directory/row['path']).resolve()
        if not p.is_relative_to(directory.resolve()):raise ValueError('checkpoint outside attempt')
        if sha(p)!=row['sha256'] or row['dtype']!=expected.dtype.name or row['shape']!=expected.shape.as_list():
            raise ValueError('checkpoint hash/type/shape mismatch')
        value=tf.io.parse_tensor(p.read_bytes(),expected.dtype)
        values.append(tf.ensure_shape(value,expected.shape))
    return tf.nest.pack_sequence_as(spec,values)

def next_look(count,policy=POLICY):
    if count>=policy['retained_max']:return None
    return min(policy['retained_max'],max(policy['retained_min'],
        (count//policy['assessment_stride']+1)*policy['assessment_stride']))

def timing_forecast(history,phase,count,adapt_count,diagnostic_seconds=0.,policy=POLICY):
    def rate(kind):
        rows=[r for r in history if r['phase']==kind and not r.get('first_compiled_call')]
        if not rows:rows=[r for r in history if not r.get('first_compiled_call')]
        return max((r['seconds']/r['count'] for r in rows[-4:]),default=None)
    ar,sr=rate('adapt'),rate('retained')
    if ar is None or sr is None:return dict(measured=False)
    if any(not math.isfinite(v) or v<=0 for v in (ar,sr)):raise ValueError('invalid measured cost')
    look=next_look(count,policy)
    remaining=max(0,(look or count)-count)
    return dict(measured=True,adapt_seconds_per_transition=ar,sample_seconds_per_transition=sr,
        sampling_measured=any(r['phase']=='retained' and not r.get('first_compiled_call') for r in history),
        next_assessment_count=look,next_assessment_seconds=1.25*(remaining*sr+diagnostic_seconds),
        maximum_adaptation_remaining_seconds=max(0,policy['warmup_max']-adapt_count)*ar if phase=='adapt' else 0.,
        reserve_factor=1.25,limitation='early descriptive forecast; dynamic tree/refinement work can change')
