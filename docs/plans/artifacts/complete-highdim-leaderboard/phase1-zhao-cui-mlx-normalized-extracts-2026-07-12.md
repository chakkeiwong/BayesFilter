# Zhao-Cui Pinned `.mlx` Normalized Code Extracts

Date: 2026-07-12

Purpose: line-addressable, read-only audit excerpts for Phase 1 P1-B. Each
block contains selected verbatim code statements from the code CDATA in
`matlab/document.xml` inside the named `.mlx` ZIP archive. Comments and
unneeded branches may be omitted. The original archive path and SHA-256 are
authoritative; this file is an audit aid and is not executable source.

Extraction form used for checking:

```text
unzip -p <archive>.mlx matlab/document.xml
extract each <![CDATA[...]]> body, then compare each selected statement below
```

## Predator-Prey Setup

Original:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/setup.mlx`

Original SHA-256:
`40f78eb59cf22365d81dbc8230d4483794b4932855ae88248dc4a693c3c81070`

```matlab
 1 function model = setup(model)
 2
 3 model.pre.theta  = [0.6; 1.2; 0.5; 0.3; 0.5; 0.5]; % true value for theta
 4 model.pre.ncons = [0.1; 1; 0.1; 0.1; 0; 0];
 5 model.theta = norminv(model.pre.theta-model.pre.ncons);
 6 model.pre.init = [50;5]; % init states
 7 model.pre.dt = 2; % step size
 8 model.pre.C = eye(model.m);
 9
10 model.pre.sigma1 = 2;
11 model.pre.sigma2 = 2;
12
13 end
```

## Predator-Prey ODE

Original:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/odefun.mlx`

Original SHA-256:
`3f4761320a4eeaec2e4307efe0fb2f76b977dd16885df30bfccf3bf9eb0acf75`

```matlab
 1 function dxdt = odefun(~, x, theta)
 2     r = theta(1, :);
 3     s = theta(2, :);
 4     u = theta(3, :);
 5     v = theta(4, :);
 6     K = theta(5, :);
 7     a = theta(6, :);
 8     dxdt1 = r.*x(1, :).*(1-x(1, :)./(90+20*K))-s.*x(1, :).*x(2, :)./(20+10*a+x(1, :));
 9     dxdt2 = u.*x(1, :).*x(2, :)./(20+10*a+x(1, :))-v.*x(2, :);
10     dxdt = [dxdt1; dxdt2];
11     dxdt = dxdt(:);
12 end
```

## Predator-Prey RK4 Step

Original:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/pp/predator_step.mlx`

Original SHA-256:
`53992fec98393509c0049e3d510e4d5710a0147e3a3ea27960abbddec218dff8`

```matlab
 1 function x_new = predator_step(model, x, theta, option, varargin)
 2 if nargin == 5
 3     delta = varargin{1};
 4 else
 5     delta = model.pre.dt;
 6 end
 7 if strcmp(option, "ode45")
 8     opt = odeset('RelTol',1e-10, 'Events', @blowup);
 9     [t, x_temp] = ode45(@(t, x) odefun(t, x, theta), [0, delta], x, opt);
10     if size(x, 2) == 1
11         x_new = x_temp(end, :)';
12     else
13         x_temp = x_temp(end, :);
14         x_new(1, :) = x_temp(1:2:end);
15         x_new(2, :) = x_temp(2:2:end);
16     end
17 elseif strcmp(option, "RK4")
18     for t = 1:20
19     x = myRK4(x, theta, delta/20);
20     end
21     x_new = x;
22 end
23
24     function x_new = myRK4(x, theta, delta)
25         fp1 = reshape(odefun(0, x, theta), 2, []);
26         fp2 = reshape(odefun(0, x + fp1.*delta/2, theta), 2, []);
27         fp3 = reshape(odefun(0, x + fp2.*delta/2, theta), 2, []);
28         fp4 = reshape(odefun(0, x + fp3*delta/2, theta), 2, []);
29         fp = (fp1 + 2*fp2 + 2*fp3 + fp4)/6;
30         x_new = x + fp * delta;
31     end
32 end
```

## Austria SIR Setup

Original:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/setup.mlx`

Original SHA-256:
`8dd1f60c973737edd0e74e81c4e95e7301428fea06a256680d3c82962b79b907`

```matlab
 1 function model = setup(model)
 2
 3 model.theta = [];
 4 model.type = 0;
 5 model.pre.theta = [.1, 18];
 6 model.pre.sigma1 = 1;
 7 model.pre.sigma2 = 10;
 8 model.pre.priormean = zeros(model.m, 1);
 9 model.pre.priormean(1:2:model.m) = 495 - model.m/2 + (1:model.m/2);
10 model.pre.priormean(2:2:model.m) = model.m/2 + 5 - (1:model.m/2);
11
12 model.pre.C = zeros(model.m/2, model.m);
13 for k = 1:model.m/2
14     model.pre.C(k, 2*k) = 1;
15 end
16
17 end
```

## Austria SIR ODE

Original:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/odefun.mlx`

Original SHA-256:
`a92d5617582e676c118f5d9b281c37960684b5b5745521e428c67aa750ee4cec`

```matlab
 1 function z = odefun(x, theta)
 2     m = size(x, 1);
 3     z = zeros(m, size(x, 2));
 4     theta1 = theta(1);
 5     theta2 = theta(2);
 6
 7     ind = zeros(9);
 8     ind(1, 1:2) = 1;
 9     ind(2, 1:4) = 1;
10     ind(3, 2:6) = 1;
11     ind(4, 2:5) = 1;
12     ind(5, [3:7, 9]) = 1;
13     ind(6, [3, 5:7]) = 1;
14     ind(7, 5:9) = 1;
15     ind(8, 7:8) = 1;
16     ind(9, 5:2:9) = 1;
17     ind = ind - eye(9);
18
19     for k = 1:9
20         z(2*k-1, :) = -theta1 .* x(2*k-1, :).* x(2*k, :) + ...
21             0.5*(sum(x(2*find(ind(k, :))-1 ,:), 1) - nnz(ind(k, :))*x(2*k-1, :));
22         z(2*k, :) = theta1 .* x(2*k-1, :).* x(2*k, :) - theta2*x(2*k, :) + ...
23             0.5*(sum(x(2*find(ind(k, :)) ,:), 1) - nnz(ind(k, :))*x(2*k, :));
24     end
25 end
```

## Austria SIR RK4 Step

Original:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/sir_step.mlx`

Original SHA-256:
`d2ceee65898b5990eecfa28078e68bf00368912063434864f342746a2f88653e`

```matlab
 1 function x_new = sir_step(x, theta, varargin)
 2 if nargin == 3
 3     delta = varargin{1};
 4 else
 5     delta = .005;
 6 end
 7 for t = 1:4
 8     x = myRK4(x, theta, delta);
 9 end
10 x_new = x;
11
12     function x_new = myRK4(x, theta, delta)
13         fp1 = odefun(x, theta);
14         fp2 = odefun(x + fp1.*delta/2, theta);
15         fp3 = odefun(x + fp2.*delta/2, theta);
16         fp4 = odefun(x + fp3*delta/2, theta);
17         fp = (fp1 + 2*fp2 + 2*fp3 + fp4)/6;
18         x_new = x + fp * delta;
19     end
20 end
```

## Check Result

On 2026-07-12, each selected statement above was compared with the CDATA
emitted from its bound archive. Added display line numbers are not part of the
source. This artifact makes no claim that the pinned author code agrees with
the paper; the predator-prey mismatch is the reason both are retained
separately.
