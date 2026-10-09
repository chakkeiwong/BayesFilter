function pdf = priorpdf(model, thetax)
pdf = mvnpdf(thetax(1:model.d, :)')'.*mvnpdf(thetax(model.d+1:model.d+model.m, :)', ...
    model.pre.init', [1^2,0;0,1^2])';
end
