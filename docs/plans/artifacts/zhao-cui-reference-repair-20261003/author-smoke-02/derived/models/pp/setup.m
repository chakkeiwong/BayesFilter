function model = setup(model)

model.pre.theta  = [0.6; 1.2; 0.5; 0.3; 0.5; 0.5]; % true value for theta
model.pre.ncons = [0.1; 1; 0.1; 0.1; 0; 0];
model.theta = norminv(model.pre.theta-model.pre.ncons);
model.pre.init = [50;5]; % init states
model.pre.dt = 2; % step size
model.pre.C = eye(model.m);

model.pre.sigma1 = 2;
model.pre.sigma2 = 2;

end
