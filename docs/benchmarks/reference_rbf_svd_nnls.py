"""Independent CPU diagnostic NNLS reference using fresh active-column SVDs."""
import numpy as np
from scipy.linalg import lstsq


def svd_nnls(design,target,stopping_factor=.01):
    rows,n=design.shape;eps=np.finfo(float).eps
    tolerance=stopping_factor*eps*np.linalg.norm(design,1)*max(rows,n)
    x=np.zeros(n);active=np.zeros(n,dtype=bool);iterations=0;minimum=1.;removed=0;peak=0
    def solve():
        nonlocal iterations,minimum,peak
        iterations+=1
        if iterations>5*n:raise RuntimeError('Fresh-SVD NNLS iteration exhaustion')
        matrix=design[:,active];peak=max(peak,int(active.sum()))
        zsmall,_,rank,singular=lstsq(matrix,target,cond=0.,lapack_driver='gelsd')
        margin=float(singular[-1]/singular[0]);minimum=min(minimum,margin)
        if rank!=active.sum() or margin<=8*eps*(rows+active.sum()):raise RuntimeError('Fresh-SVD NNLS active rank failure')
        z=np.zeros(n);z[active]=zsmall
        return z
    gradient=design.T@(target-design@x)
    while np.any((~active)&(gradient>tolerance)):
        column=int(np.argmax(np.where(active,-np.inf,gradient)));active[column]=True;z=solve()
        while np.any(active&(z<0)):
            negative=active&(z<0);fraction=np.full(n,np.inf);fraction[negative]=x[negative]/(x[negative]-z[negative])
            alpha=float(np.min(fraction));drop=negative&(fraction==alpha)
            x=(1-alpha)*x+alpha*z;x[drop]=0.;active[drop]=False;removed+=int(drop.sum())
            if not active.any():z=np.zeros(n);break
            z=solve()
        x=z
        if np.any(x<0) or not np.all(np.isfinite(x)):raise RuntimeError('Fresh-SVD NNLS feasibility failure')
        gradient=design.T@(target-design@x)
    return x,dict(iterations=iterations,minimum_svd_margin=minimum,peak_active=peak,removed=removed,gradient_tolerance=float(tolerance))
