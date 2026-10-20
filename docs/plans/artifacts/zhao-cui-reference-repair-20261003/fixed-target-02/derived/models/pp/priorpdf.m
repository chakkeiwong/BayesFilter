function pdf = priorpdf(model, x)
pdf = mvnpdf(x', model.pre.init', eye(model.m))';
end
