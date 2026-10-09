function x_new = sir_step(x, theta, varargin)
if nargin == 3
    delta = varargin{1};
else
    delta = .005;
end
for t = 1:4
    x = myRK4(x, theta, delta);
end
x_new = x;

    function x_new = myRK4(x, theta, delta)
        fp1 = odefun(x, theta);
        fp2 = odefun(x + fp1.*delta/2, theta);
        fp3 = odefun(x + fp2.*delta/2, theta);
        fp4 = odefun(x + fp3*delta/2, theta);
        fp = (fp1 + 2*fp2 + 2*fp3 + fp4)/6;
        x_new = x + fp * delta;
    end
end
