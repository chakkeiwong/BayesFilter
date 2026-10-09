function Y = ob_process(model, thetax, ~)
Y = model.pre.C * thetax(model.d+1:end, :) + model.pre.sigma2 *randn(model.n, size(thetax(model.d+1:end, :), 2));
end
