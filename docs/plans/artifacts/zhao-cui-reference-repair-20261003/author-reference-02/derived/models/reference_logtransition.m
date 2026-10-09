function value = reference_logtransition(model, thetax, t)
previous = thetax(model.d+model.m+1:model.d+2*model.m,:);
if strcmp(model.name, 'pp')
    if model.d == 0
        theta = repmat(model.pre.theta,1,size(thetax,2));
    else
        theta = bsxfun(@plus,model.pre.ncons,normcdf(thetax(1:model.d,:)));
    end
    mu = predator_step(model,previous,theta,'RK4');
else
    mu = sir_step(previous,model.pre.theta);
end
residual = (thetax(model.d+1:model.d+model.m,:)-mu)/model.pre.sigma1;
value = -0.5*(model.m*log(2*pi) + 2*model.m*log(model.pre.sigma1) + sum(residual.^2,1));
end
