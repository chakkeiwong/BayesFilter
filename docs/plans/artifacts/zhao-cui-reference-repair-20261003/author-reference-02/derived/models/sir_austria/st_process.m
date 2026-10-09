function X_new = st_process(model, thetax, ~)
X_new = sir_step(thetax(model.d+1:model.d+model.m, :), model.pre.theta) + model.pre.sigma1*randn(model.m, size(thetax, 2));
X_new(1:2:model.m, :) = max(X_new(1:2:model.m, :), 0);
end
