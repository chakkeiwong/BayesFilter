function pdf = like(model, thetax, t)
y = model.Y(:, t);
% n = size(X,2);
% pdf = zeros(1,n);
% d2 = size(y,1);
% for k = 1:n
%     if any(isnan(X(:, k)))
%         pdf(k) = 0;
%     else
%         pdf(k) = mvnpdf(y, (model.C * X(model.td+1:model.td+model.dimension, k)), model.sigma2^2 * eye(d2));
%     end
% 
% end
pdf = mvnpdf(y', (model.pre.C*thetax(model.d+1:model.d+model.m, :))', model.pre.sigma2^2 * eye(model.n))';
pdf(isnan(pdf)) = 0;
end
