"""WALNUTS-R2P: paper sections 3.1/3.3, eqs 13–14; author macro parity.

The exponent c denotes 2**c microsteps, not c microsteps. Endpoint energy
criterion follows Remark 13. Runtime implementation uses TensorFlow only.
"""
from typing import NamedTuple
from .core import Kernel, Span

RSpan = NamedTuple('RSpan', [(name,object) for name in Span._fields]+[('bc',object),('fc',object)])

class MacroResult(NamedTuple):
    point: object
    forward: object
    backward: object
    exponent: object
    log_ratio: object
    forward_evaluations: object
    backward_evaluations: object
    finite: object
    threshold_met: object
    seed: object
    adam: object

class R2PKernel(Kernel):
    def march(self,start,inv,h,exponent,active):
        tf=self.tf
        count=tf.bitwise.left_shift(tf.ones_like(exponent),exponent)
        step=h/tf.cast(count,self.dtype)
        def body(i,p,finite):
            live=active & (i<count)
            mom=p.p+.5*step[:,None]*p.g
            q=self.choose(live,p.q+step[:,None]*inv*mom,start.q)
            lp,g=self.value_gradient(q)
            mom+=.5*step[:,None]*g
            nxt=self.point(q,mom,g,lp,inv)
            finite &= ~live | (self.health(q,lp,g) & tf.math.is_finite(nxt.joint))
            return i+1,self.choose(live,nxt,p),finite
        _,out,finite=tf.while_loop(lambda i,p,f:tf.reduce_any(active & (i<count)),body,
            (tf.constant(0),start,tf.ones([self.c],tf.bool)),parallel_iterations=1)
        return out,finite,count

    def critical(self,start,inv,h,active,upper):
        tf=self.tf
        upper=tf.broadcast_to(tf.convert_to_tensor(upper,tf.int32),[self.c])
        def body(c,live,out,chosen,met,finite,ev):
            live &= c<=upper
            trial,fin,cost=self.march(start,inv,h,tf.fill([self.c],c),live)
            good=live & fin & (tf.abs(trial.joint-start.joint)<self.cfg.max_error)
            return (c+1,live & ~good,self.choose(live,trial,out),
                tf.where(live,c,chosen),met|good,tf.where(live,fin,finite),
                ev+tf.where(live,cost,self.ints()))
        _,_,out,chosen,met,finite,ev=tf.while_loop(
            lambda c,live,*_: tf.reduce_any(live & (c<=upper)),body,
            (tf.constant(0),active,start,tf.maximum(upper,0),tf.zeros([self.c],tf.bool),
             tf.ones([self.c],tf.bool),self.ints()),parallel_iterations=1)
        return out,chosen,met,finite,ev

    def r2p_macro(self,start,inv,h,active,seed,adam,adapt=False):
        tf,c=self.tf,self.cfg
        # Source fallback at the cap defines the selector even if tolerance fails.
        trial,If,met,finite,evf=self.critical(start,inv,h,active,c.r2p_max_exponent)
        u,seed=self.random(seed,active)
        finer=active & (u>=c.r2p_probability)
        chosen=If+tf.cast(finer,tf.int32)
        refined,fin,cost=self.march(start,inv,h,chosen,finer)
        trial=self.choose(finer,refined,trial);finite=tf.where(finer,fin,finite)
        evf+=tf.where(finer,cost,self.ints())
        # Author optimization: the baseline endpoint already witnesses If.
        # The randomized finer endpoint needs a new search through maxC.
        reverse=trial._replace(p=-trial.p)
        upper=tf.where(finer,c.r2p_max_exponent,If-1)
        _,searched,back_met,back_finite,evb=self.critical(reverse,inv,h,active,upper)
        Ib=tf.where(finer,searched,tf.where(back_met,searched,If))
        logp=tf.math.log(self.f(c.r2p_probability));log1p=tf.math.log(self.f(1-c.r2p_probability))
        log_forward=tf.where(finer,log1p,logp)
        log_backward=tf.where(chosen==Ib,logp,
            tf.where(chosen==Ib+1,log1p,self.f(float('-inf'))))
        ratio=log_backward-log_forward
        ratio=tf.where(finite,ratio,self.f(float('-inf')))
        if adapt:
            base,fin,_=self.march(start,inv,h,self.ints(),active)
            alpha=tf.exp(-tf.abs(base.joint-start.joint))
            alpha=tf.where(fin & tf.math.is_finite(alpha),alpha,self.zeros())
            adam=self.adam_observe(adam,alpha,active)
            evf+=tf.cast(active,tf.int32)
        return MacroResult(trial,If,Ib,chosen,ratio,evf,evb,finite,met,seed,adam)

    def span(self,p):
        return RSpan(*super().span(p),self.zeros(),self.zeros())

    def grow_leaf(self,old,forward,inv,eps,minimum,live,seed,adam,adapt):
        tf=self.tf
        h=eps
        if self.cfg.r2p_jitter:
            u,seed=self.random(seed,live)
            h=eps*(1+self.cfg.r2p_jitter*(2*u-1))
        r=self.r2p_macro(self.edge(old,forward),inv,tf.where(forward,h,-h),live,seed,adam,adapt)
        correction=self.choose(forward,old.fc,old.bc)+r.log_ratio
        span=self.span(r.point)._replace(weight=r.point.joint+correction,bc=correction,fc=correction)
        # Zero probability is a zero-weight geometric endpoint, not a new state
        # with an uncorrected weight, and not a blanket rejection of its subtree.
        return span,live & r.finite,live & ~tf.math.is_finite(r.log_ratio),r.forward_evaluations+r.backward_evaluations,r.seed,r.adam

    def combine(self,old,new,forward,seed,active,metropolis):
        tf=self.tf
        # A zero-weight subtree still has geometric endpoints. Avoid undefined
        # (-inf)-(-inf) selection when both candidate weights are zero.
        empty=tf.math.is_inf(old.weight)&(old.weight<0)&tf.math.is_inf(new.weight)&(new.weight<0)
        ow=tf.where(empty,self.zeros(),old.weight);nw=tf.where(empty,self.zeros(),new.weight)
        total=tf.math.reduce_logsumexp(tf.stack([ow,nw]),axis=0)
        u,seed=self.random(seed,active)
        select=~empty & (tf.math.log(u)<nw-(ow if metropolis else total))
        total=tf.where(empty,self.f(float('-inf')),total)
        b,f=self.choose(forward,old,new),self.choose(forward,new,old)
        chosen=self.choose(select,new,old)
        return RSpan(b.bq,b.bp,b.bg,b.bj,f.fq,f.fp,f.fg,f.fj,chosen.q,chosen.g,chosen.lp,total,b.bc,f.fc),seed

    def stack_init(self,old):
        if self.cfg.execution_layout=='fieldwise':
            return RSpan(*(self.tf.zeros([self.cfg.max_depth+1]+x.shape.as_list(),x.dtype) for x in old))
        return self.tf.zeros([self.cfg.max_depth+1,self.c,8*self.d+6],self.dtype)

    def stack_read(self,stack,level):
        if self.cfg.execution_layout=='fieldwise':return RSpan(*(x[level] for x in stack))
        parts=self.tf.split(stack[level],[self.d,self.d,self.d,1,self.d,self.d,self.d,1,self.d,self.d,1,1,1,1],-1)
        return RSpan(*(x[:,0] if i in (3,7,10,11,12,13) else x for i,x in enumerate(parts)))

    def stack_write(self,stack,level,span):
        tf=self.tf;index=tf.reshape(level,[1,1])
        if self.cfg.execution_layout=='fieldwise':
            return RSpan(*(tf.tensor_scatter_nd_update(x,index,y[None]) for x,y in zip(stack,span)))
        return tf.tensor_scatter_nd_update(stack,index,self.pack_span(span)[None])
