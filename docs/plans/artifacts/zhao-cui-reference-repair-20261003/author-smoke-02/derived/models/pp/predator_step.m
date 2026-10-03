function x_new = predator_step(model, x, theta, option, varargin)
%{
calculate the states in next step for predator model
input is x: the current states
         dt: fixed step length
         theta: parameters
         option: 1 for cts case using ode solver. 2 for Euler scheme
%}
if nargin == 5
    delta = varargin{1};
else
    delta = model.pre.dt;
end
if strcmp(option, "ode45")
    opt = odeset('RelTol',1e-10, 'Events', @blowup);
    [t, x_temp] = ode45(@(t, x) odefun(t, x, theta), [0, delta], x, opt);
    if size(x, 2) == 1
        x_new = x_temp(end, :)';
    else
        x_temp = x_temp(end, :);
        x_new(1, :) = x_temp(1:2:end);
        x_new(2, :) = x_temp(2:2:end);
    end
elseif strcmp(option, "RK4")
    for t = 1:20
    x = myRK4(x, theta, delta/20);
    end
    x_new = x;
end

    function x_new = myRK4(x, theta, delta)
        fp1 = reshape(odefun(0, x, theta), 2, []);
        fp2 = reshape(odefun(0, x + fp1.*delta/2, theta), 2, []);
        fp3 = reshape(odefun(0, x + fp2.*delta/2, theta), 2, []);
        fp4 = reshape(odefun(0, x + fp3*delta/2, theta), 2, []);
        fp = (fp1 + 2*fp2 + 2*fp3 + fp4)/6;
        x_new = x + fp * delta;
    end
end
