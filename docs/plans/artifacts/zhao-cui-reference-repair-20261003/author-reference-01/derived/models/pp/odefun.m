function dxdt = odefun(~, x, theta)
    r = theta(1, :);
    s = theta(2, :);
    u = theta(3, :);
    v = theta(4, :);
    K = theta(5, :);
    a = theta(6, :);
    dxdt1 = r.*x(1, :).*(1-x(1, :)./(90+20*K))-s.*x(1, :).*x(2, :)./(20+10*a+x(1, :));
    dxdt2 = u.*x(1, :).*x(2, :)./(20+10*a+x(1, :))-v.*x(2, :);
    dxdt = [dxdt1; dxdt2];
    dxdt = dxdt(:);
end
