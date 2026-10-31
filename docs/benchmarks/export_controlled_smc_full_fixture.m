% Diagnostic extraction only: never reruns or modifies the author algorithm.
args=argv(); input=args{1}; output=args{2};
load(input); H=length(y); fixture=struct('y',y','ntrials',50,'alpha',.99,'variance',.11,'rounds',{{}});
for j=1:rows(result)
  policy=result{j,1}; smc=result{j,2};
  coeff=[reshape(policy.A,H,1),policy.b,policy.c];
  fixture.rounds{j}=struct('policy',coeff,'x',squeeze(smc.Trajectory)',
    'ancestors',smc.Ancestry'-1,'log_incremental_weight',smc.LogIncrementalWeight',
    'logZ',smc.LogNormConst,'ess',smc.ESS);
end
save('-mat7-binary',output,'fixture');
