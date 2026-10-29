function z = odefun(x, theta)
    m = size(x, 1);
    z = zeros(m, size(x, 2));
    theta1 = theta(1);
    theta2 = theta(2);

    ind = zeros(9);
    ind(1, 1:2) = 1;
    ind(2, 1:4) = 1;
    ind(3, 2:6) = 1;
    ind(4, 2:5) = 1;
    ind(5, [3:7, 9]) = 1;
    ind(6, [3, 5:7]) = 1;
    ind(7, 5:9) = 1;
    ind(8, 7:8) = 1;
    ind(9, 5:2:9) = 1;
    ind = ind - eye(9);

    for k = 1:9
        z(2*k-1, :) = -theta1 .* x(2*k-1, :).* x(2*k, :) + ...
            0.5*(sum(x(2*find(ind(k, :))-1 ,:), 1) - nnz(ind(k, :))*x(2*k-1, :));
        z(2*k, :) = theta1 .* x(2*k-1, :).* x(2*k, :) - theta2*x(2*k, :) + ...
            0.5*(sum(x(2*find(ind(k, :)) ,:), 1) - nnz(ind(k, :))*x(2*k, :));
    end
end
