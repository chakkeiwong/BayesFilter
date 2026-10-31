"""Independent diagnostic proposal construction; never sampler/tuner code."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def objective_gradient(log_target, point, step=1.e-5):
    point = np.asarray(point, dtype=float)
    directions = step*np.eye(len(point))
    values = -np.asarray(log_target(np.concatenate(
        (point[None], point+directions, point-directions))))
    if values.shape != (1+2*len(point),) or not np.isfinite(values).all():
        raise ValueError('nonfinite independent optimizer objective')
    return float(values[0]), (values[1:1+len(point)]-values[1+len(point):])/(2*step)


def checked_curvature(log_target, point):
    point = np.asarray(point, dtype=float)
    dimension = len(point)
    def hessian(step):
        directions = step*np.eye(dimension)
        points = [point]
        for i in range(dimension):
            for j in range(dimension):
                points.extend((point+directions[i]+directions[j],
                    point+directions[i]-directions[j],
                    point-directions[i]+directions[j],
                    point-directions[i]-directions[j]))
        values = -np.asarray(log_target(np.asarray(points)))
        if not np.isfinite(values).all():
            raise ValueError('nonfinite independent curvature')
        values = values[1:].reshape(dimension,dimension,4)
        result = (values[:,:,0]-values[:,:,1]-values[:,:,2]+values[:,:,3])/(4*step*step)
        return .5*(result+result.T)
    coarse, fine = hessian(1.e-3), hessian(5.e-4)
    discrepancy = np.linalg.norm(fine-coarse)/np.linalg.norm(fine)
    eigenvalues = np.linalg.eigvalsh(fine)
    if not np.isfinite(discrepancy) or discrepancy > .01 or np.min(eigenvalues) <= 0:
        raise ValueError('independent curvature is nonpositive or step-size unstable')
    factor = np.linalg.cholesky(np.linalg.inv(fine))
    return factor, dict(relative_hessian_discrepancy=float(discrepancy),
                        curvature_eigenvalues=eigenvalues.tolist())


def local_proposal(log_target, center, *, on_start=None):
    center = np.asarray(center, dtype=float)
    if center.shape != (9,) or not np.isfinite(center).all():
        raise ValueError('nine finite standardized coordinates required')
    starts = [center.copy(), np.zeros(9)]
    for frequency in (-.8,-.4,0.,.4,.8):
        start = center.copy(); start[3] = frequency/.35; starts.append(start)
    candidates, receipts = [], []
    for index, start in enumerate(starts):
        result = minimize(lambda x: objective_gradient(log_target,x),start,
            jac=True,method='BFGS',options=dict(gtol=1.e-5,maxiter=300))
        _, residual = objective_gradient(log_target,result.x,step=5.e-6)
        receipt = dict(index=index,start=start.tolist(),point=result.x.tolist(),
            objective=float(result.fun),optimizer_success=bool(result.success),
            message=str(result.message),iterations=int(result.nit),
            max_gradient=float(np.max(np.abs(residual))))
        if receipt['max_gradient'] <= 1.e-3:
            try:
                factor, curvature = checked_curvature(log_target,result.x)
                receipt.update(curvature=curvature,local_geometry_usable=True)
                candidates.append((float(result.fun),result.x.copy(),factor))
            except ValueError as exc:
                receipt.update(local_geometry_usable=False,curvature_error=str(exc))
        else:
            receipt['local_geometry_usable'] = False
        receipts.append(receipt)
        if on_start is not None: on_start(receipt)
    unique=[]
    for candidate in sorted(candidates,key=lambda x:x[0]):
        if all(np.linalg.norm(np.linalg.solve(old[2],candidate[1]-old[1])) >= 1.
               for old in unique):
            unique.append(candidate)
    if not unique:
        raise ValueError('no independently stationary positive-curvature proposal center')
    modes=unique[:3]
    centers=[np.zeros(9)]; factors=[np.eye(9)]
    for index in range(7):
        _, mode, factor=modes[index%len(modes)]
        width=(1.,2.,3.)[(index//len(modes))%3]
        centers.append(mode); factors.append(width*factor)
    proposal=dict(centers=np.asarray(centers).tolist(),factors=np.asarray(factors).tolist(),
        weights=[1/8]*8,evidence_role='independent_reference_proposal_only')
    guards=[dict(center=m[1].tolist(),factor=(2*m[2]).tolist()) for m in modes]
    return proposal, guards, dict(starts=receipts,distinct_stationary_modes=len(unique),
        retained_modes=len(modes),global_mode_coverage_proved=False)


def preserve_local_coverage(proposal, guards):
    """Keep broad local components after EM; full mixture weights remain exact."""
    result=dict(proposal,centers=[list(x) for x in proposal['centers']],
                factors=[[list(row) for row in a] for a in proposal['factors']])
    if not 1 <= len(guards) <= 3:
        raise ValueError('one to three independently fitted local guards required')
    for index, guard in enumerate(guards):
        result['centers'][-1-index]=guard['center']
        result['factors'][-1-index]=guard['factor']
    result['broad_local_components']=len(guards)
    return result
