function X = mvnrnd(mu, Sigma, N)
mu = mu(:)';
d = numel(mu);
if ~isequal(size(Sigma), [d, d]) || any(~isfinite(Sigma(:)))
    error('reference mvnrnd: invalid covariance');
end
[R, status] = chol(Sigma);
if status ~= 0, error('reference mvnrnd: covariance must be positive definite'); end
X = bsxfun(@plus, randn(N, d)*R, mu);
end
