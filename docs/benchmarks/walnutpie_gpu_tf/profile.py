"""Bounded GPU engineering profiler; no scientific or speed promotion."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from .worker import setup, timed, write, graph_report
from .api import make_functions
from .core import Config

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tf,report=setup('gpu')
    with tf.device('/GPU:0'):
        cfg=Config(4,10,warmup=4,draws=8)
        f=make_functions(tf,lambda q:-.5*tf.reduce_sum(q*q,-1),cfg)
        q=tf.reshape(tf.sin(tf.cast(tf.range(40),tf.float64)),[4,10])*.3
        seeds=tf.constant([[201+i,0] for i in range(4)],tf.int32)
        result,first=timed(tf,f.fit,q,seeds)
        report['first_call_seconds']=first
        report['steady_seconds']=[]
        for _ in range(7):result,t=timed(tf,f.fit,q,seeds);report['steady_seconds'].append(t)
        report['draws']=result.sampling.draws.numpy().tolist()
        report['graph']=graph_report(f.fit,(q,seeds),a.output/'fit.hlo.txt')
        from tensorflow.python.eager import context
        context.async_wait()
        tf.profiler.experimental.start(str(a.output/'profile'),options=tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2,python_tracer_level=0,device_tracer_level=1))
        try:
            with tf.profiler.experimental.Trace('walnutpie_full_fit',step_num=0,_r=1):
                f.fit(q,seeds);context.async_wait()
        finally:tf.profiler.experimental.stop()
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    gpu=Counter();host=Counter();copies=[]
    for path in (a.output/'profile').rglob('*.xplane.pb'):
        s=xplane_pb2.XSpace();s.ParseFromString(path.read_bytes())
        for plane in s.planes:
            for line in plane.lines:
                for event in line.events:
                    name=plane.event_metadata[event.metadata_id].name
                    if 'gpu' in plane.name.lower():gpu[name]+=1
                    else:host[name]+=1
                    if 'memcpy' in name.lower():copies.append(dict(plane=plane.name,name=name))
    if not gpu:raise RuntimeError('missing GPU events')
    report.update(gpu_events=dict(gpu),host_top_events=host.most_common(30),copies=copies,
        profile_scope='TF profiler disables command buffers; profiler shows fallback scheduling, unprofiled timings retain requested flags',
        allocator=tf.config.experimental.get_memory_info('GPU:0'))
    write(a.output/'profile.json',report);print(json.dumps({'gpu_events':sum(gpu.values()),'copies':len(copies)}))

if __name__=='__main__':main()
