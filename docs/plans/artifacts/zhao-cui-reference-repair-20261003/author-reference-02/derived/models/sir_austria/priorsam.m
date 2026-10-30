function sams = priorsam(model, N)
sams = mvnrnd(model.pre.priormean', 1e0*eye(model.m), N)';
end
