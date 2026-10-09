function value = reference_loglike(model, thetax, t)
mu = model.pre.C*thetax(model.d+1:model.d+model.m,:);
residual = bsxfun(@minus,model.Y(:,t),mu)/model.pre.sigma2;
value = -0.5*(model.n*log(2*pi) + 2*model.n*log(model.pre.sigma2) + sum(residual.^2,1));
end
