function sams = priorsam(model, N)
sams = [randn(model.d, N); mvnrnd(model.pre.init, [1^2,0;0,1^2], N)'];
end
