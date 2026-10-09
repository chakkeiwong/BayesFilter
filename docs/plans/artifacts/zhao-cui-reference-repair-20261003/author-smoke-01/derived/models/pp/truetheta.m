function model = truetheta(model)

model.dimension = 2;
model.td = 6; % dimension of parameters
model.sigma1 = 5; % process error
model.sigma2 = 1; % observation error
model.init = [50;5]; % init states
model.dt = 4; % step size
model.C = eye(model.dimension);
model.theta  = [0.6; 1.2; 0.5; 0.3; 0.5; 0.5]; % true value for theta
model.ncons = [0.1; 1; 0.1; 0.1; 0; 0];
model.theta_transformed = norminv(model.theta-model.ncons);

end
