function pdf = transition(model, thetax, ~)
% n = size(X, 2);
% pdf = zeros(1, n);
% 
% for k = 1:n
%     temp = predator_step(X(model.td+model.dimension+1:model.td+2*model.dimension, k), model.dt, model.ncons + normcdf(X(1:model.td, k)));
%     if any(isnan(temp))
%         pdf(k) = 0;
%     elseif temp < 1e100
%     pdf(k) = mvnpdf(X(model.td+1:model.td+model.dimension, k)', temp', model.sigma1^2 * eye(model.dimension));
%     else
%         pdf(k) = 0;
%     end
% end
temp = predator_step(model, thetax(model.d+model.m+1 : model.d+2*model.m, :), ...
    model.pre.ncons + normcdf(thetax(1:model.d, :)), "RK4");
pdf = mvnpdf(thetax(model.d+1 : model.d+model.m, :)', temp', model.pre.sigma1^2*eye(model.m))';
end
