function model = setup(model)

model.theta = [];
model.type = 0;
model.pre.theta = [.1, 18];
model.pre.sigma1 = 1;
model.pre.sigma2 = 10;
model.pre.priormean = zeros(model.m, 1);
model.pre.priormean(1:2:model.m) = 495 - model.m/2 + (1:model.m/2);
model.pre.priormean(2:2:model.m) = model.m/2 + 5 - (1:model.m/2);

model.pre.C = zeros(model.m/2, model.m);
for k = 1:model.m/2
    model.pre.C(k, 2*k) = 1;
end

end
