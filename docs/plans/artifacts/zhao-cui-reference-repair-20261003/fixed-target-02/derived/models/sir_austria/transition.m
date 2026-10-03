function pdf = transition(model, thetax, ~)
% n = size(thetax, 2);
% pdf = zeros(1, n);
% 
% for k = 1:n
%     temp = lrz_step(thetax(model.dimension + 1 : 2 * model.dimension, k), model.dt, model.F);
%     if any(isnan(temp))
%         pdf(k) = 0;
%     elseif all(temp < 1e100)
%     pdf(k) = mvnpdf(thetax(model.td+1:model.td+model.dimension, k)', temp', model.sigma1^2 * eye(model.dimension));
%     else
%         pdf(k) = 0;
%     end
% end

temp = sir_step(thetax(model.d+model.m+1 : model.d+2*model.m, :), model.pre.theta);
pdf = mvnpdf(thetax(model.d+1 : model.d+model.m, :)', temp', model.pre.sigma1^2*eye(model.m))';

end
