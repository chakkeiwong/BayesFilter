function pdf = priorpdf(model, thetax)
pdf = mvnpdf(thetax(model.d + 1 : model.d + model.m, :)', model.pre.priormean', 1e0*eye(model.m))';
end
