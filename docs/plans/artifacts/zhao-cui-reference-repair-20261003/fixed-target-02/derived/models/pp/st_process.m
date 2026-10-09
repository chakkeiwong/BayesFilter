function x = st_process(model, previous, t)
theta = repmat(model.pre.theta, 1, size(previous,2));
x = predator_step(model,previous,theta,'RK4') + model.pre.sigma1*randn(size(previous));
end
