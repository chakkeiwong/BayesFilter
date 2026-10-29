"""Experimental split execution: CPU/XLA tree, device/XLA integration leaf.

Host code schedules compiled calls and copies packed endpoint buffers only.
Model arrays stay resident on the integration device. This is deliberately
not described as one fused multi-device XLA executable. No NumPy or callbacks.
Fixed-kernel sampling only; existing core/R2P primitives remain authoritative.
"""
from typing import NamedTuple
from .api import make_kernel, state_spec, Run
from .core import Adam, Span, Transition
from .r2p import RSpan


class Cursor(NamedTuple):
    level: object
    index: object
    live: object
    old: object
    last: object
    stack: object
    forward: object
    seed: object
    adam: object
    depth: object
    evaluations: object
    macros: object
    failed: object
    reverse_failed: object
    inverse_mass: object
    epsilon: object
    minimum: object
    done: object


def make_hybrid(tf, model, config, model_signature=(), leaf_device='/GPU:0'):
    if config.execution_layout != 'fieldwise':
        raise ValueError('hybrid experiment uses fieldwise source tree only')
    c=config; k=make_kernel(tf,model,c); d=c.dimension; n=c.chains
    span_type=RSpan if c.variant=='r2p' else Span
    widths=[d,d,d,1,d,d,d,1,d,d,1,1]+([1,1] if c.variant=='r2p' else [])
    sw=sum(widths)
    def unpack(x):
        parts=tf.split(x,widths,axis=-1)
        return span_type(*(v[:,0] if w==1 and i not in (0,1,2,4,5,6,8,9) else v
                           for i,(v,w) in enumerate(zip(parts,widths))))
    vector=tf.TensorSpec([n],tf.float64);matrix=tf.TensorSpec([n,d],tf.float64)
    integer=tf.TensorSpec([n],tf.int32);mask=tf.TensorSpec([n],tf.bool)
    scalar=tf.TensorSpec([],tf.int32);flag=tf.TensorSpec([],tf.bool)
    span=span_type(*(matrix if i in (0,1,2,4,5,6,8,9) else vector for i in range(len(widths))))
    stack=span_type(*(tf.TensorSpec([c.max_depth+1]+v.shape.as_list(),tf.float64) for v in span))
    adam=Adam(*(vector for _ in range(6)))
    cursor=Cursor(scalar,scalar,mask,span,span,stack,mask,tf.TensorSpec([n,2],tf.int32),adam,
                  integer,integer,integer,mask,mask,matrix,vector,integer,flag)
    fps=tf.TensorSpec([n,sw+d+1+6],tf.float64);ips=tf.TensorSpec([n,5],tf.int32)
    fpr=tf.TensorSpec([n,sw+6],tf.float64);ipr=tf.TensorSpec([n,5],tf.int32)

    @tf.function(input_signature=[state_spec(tf,c)],jit_compile=True,autograph=False)
    def begin(state):
        z,seed=k.random(state.seed,k.active(),normal=True)
        old=k.span(k.point(state.q,z/tf.sqrt(state.inverse_mass),state.g,state.lp,state.inverse_mass))
        u,seed=k.random(seed,k.active())
        return Cursor(tf.constant(0),tf.constant(0),k.active(),old,old,k.stack_init(old),u<.5,seed,state.adam,
            k.ints(),k.ints(),k.ints(),tf.zeros([n],tf.bool),tf.zeros([n],tf.bool),
            state.inverse_mass,state.epsilon,state.minimum,tf.constant(False))

    @tf.function(input_signature=[cursor],jit_compile=True,autograph=False)
    def request(s):
        fp=tf.concat([k.pack_span(s.last),s.inverse_mass,s.epsilon[:,None],tf.stack(s.adam,-1)],-1)
        ip=tf.stack([tf.cast(s.forward,tf.int32),tf.cast(s.live,tf.int32),s.minimum,s.seed[:,0],s.seed[:,1]],-1)
        return fp,ip

    @tf.function(input_signature=[fps,ips,*model_signature],jit_compile=True,autograph=False)
    def leaf(fp,ip,*data):
        local=make_kernel(tf,lambda q:model(q,*data),c)
        old=unpack(fp[:,:sw]);inv=fp[:,sw:sw+d];eps=fp[:,sw+d]
        a=Adam(*(fp[:,sw+d+1+i] for i in range(6)))
        nxt,ok,bad,cost,seed,a=local.grow_leaf(old,ip[:,0]>0,inv,eps,ip[:,2],ip[:,1]>0,ip[:,3:5],a,False)
        return tf.concat([local.pack_span(nxt),tf.stack(a,-1)],-1),tf.stack([tf.cast(ok,tf.int32),
            tf.cast(bad,tf.int32),cost,seed[:,0],seed[:,1]],-1)

    @tf.function(input_signature=[cursor,fpr,ipr],jit_compile=True,autograph=False)
    def advance(s,fp,ip):
        nxt=unpack(fp[:,:sw]);a=Adam(*(fp[:,sw+i] for i in range(6)));seed=ip[:,3:5]
        ok=ip[:,0]>0;live=s.live & ok
        failed=s.failed | (s.live & ~ok);revfail=s.reverse_failed | (ip[:,1]>0)
        ev=s.evaluations+ip[:,2];macros=s.macros+tf.cast(s.live,tf.int32)
        def cond(level,span,live,seed):
            return (level<s.level)&(tf.math.floormod(s.index+1,tf.bitwise.left_shift(1,level+1))==0)&tf.reduce_any(live)
        def body(level,span,live,seed):
            prev=k.stack_read(s.stack,level)
            live &= ~k.uturn(prev,span,s.forward,s.inverse_mass)
            combined,seed=k.combine(prev,span,s.forward,seed,live,False)
            return level+1,combined,live,seed
        carry,nxt,live,seed=tf.while_loop(cond,body,(tf.constant(0),nxt,live,seed),parallel_iterations=1)
        stack=k.stack_write(s.stack,carry,nxt);index=s.index+1
        pending=s._replace(index=index,live=live,last=nxt,stack=stack,seed=seed,adam=a,
            evaluations=ev,macros=macros,failed=failed,reverse_failed=revfail)
        def close_subtree():
            turn=k.uturn(s.old,nxt,s.forward,s.inverse_mass)
            combined,rng=k.combine(s.old,nxt,s.forward,seed,live,True)
            old=k.choose(live,combined,s.old);outer_live=live & ~turn
            level=s.level+1;done=(level>=c.max_depth)|~tf.reduce_any(outer_live)
            closed=pending._replace(level=level,index=tf.constant(0),live=outer_live,old=old,last=old,
                stack=k.stack_init(old),seed=rng,depth=s.depth+tf.cast(live,tf.int32),done=done)
            def next_direction():
                u,rng=k.random(closed.seed,closed.live)
                return closed._replace(forward=u<.5,seed=rng)
            return tf.cond(done,lambda:closed,next_direction)
        return tf.cond((index>=tf.bitwise.left_shift(1,s.level))|~tf.reduce_any(live),close_subtree,lambda:pending)

    @tf.function(input_signature=[state_spec(tf,c),cursor],jit_compile=True,autograph=False)
    def finish(state,s):
        t=Transition(s.old.q,s.old.lp,s.old.g,s.seed,s.adam,s.depth,s.evaluations,s.macros,s.failed,s.reverse_failed)
        valid=state.valid & k.health(t.q,t.lp,t.g) & tf.reduce_all(t.seed>=0,-1)
        return state._replace(q=t.q,lp=t.lp,g=t.g,seed=t.seed,adam=t.adam,valid=valid),t

    class Driver:
        def __init__(self):
            self.functions={'begin':begin,'request':request,'leaf':leaf,'advance':advance,'finish':finish}
            self.signature=dict(cursor=cursor,send_float=fps,send_int=ips,return_float=fpr,return_int=ipr)
            self.last_round_trips=0
            self.bytes_per_round_trip=n*((sw+d+1+6+sw+6)*8+10*4)
            self.graph_examples={}
        def sample(self,state,count,*data):
            arrays=[[] for _ in range(7)];trips=0
            for _ in range(count):
                with tf.device('/CPU:0'):s=begin(state)
                self.graph_examples.setdefault('begin',(state,))
                while not bool(s.done.numpy()):
                    with tf.device('/CPU:0'):fp,ip=request(s)
                    self.graph_examples.setdefault('request',(s,))
                    with tf.device(leaf_device):
                        gf,gi=tf.identity(fp),tf.identity(ip)
                        rf,ri=leaf(gf,gi,*data)
                    self.graph_examples.setdefault('leaf',(gf,gi,*data))
                    with tf.device('/CPU:0'):
                        cf,ci=tf.identity(rf),tf.identity(ri)
                        old=s;s=advance(s,cf,ci)
                    self.graph_examples.setdefault('advance',(old,cf,ci));trips+=1
                with tf.device('/CPU:0'):new,t=finish(state,s)
                self.graph_examples.setdefault('finish',(state,s));state=new
                for a,x in zip(arrays,(t.q,t.lp,t.depth,t.evaluations,t.macros,t.failed,t.reverse_failed)):a.append(x)
            with tf.device('/CPU:0'):out=Run(state,*(tf.stack(a) for a in arrays))
            self.last_round_trips=trips
            return out
    return Driver()
