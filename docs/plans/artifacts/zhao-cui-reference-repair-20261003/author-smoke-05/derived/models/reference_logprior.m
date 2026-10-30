function value = reference_logprior(model, thetax)
if strcmp(model.name, 'pp'), mu = model.pre.init; else, mu = model.pre.priormean; end
residual = bsxfun(@minus, thetax(model.d+1:model.d+model.m,:), mu);
value = -0.5*(model.m*log(2*pi) + sum(residual.^2,1));
if model.d > 0
    value = value - 0.5*(model.d*log(2*pi) + sum(thetax(1:model.d,:).^2,1));
end
end
