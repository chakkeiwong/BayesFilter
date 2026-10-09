#!/usr/bin/env python3
"""Independent CPU/Octave replication diagnostic; never an admitted runtime.

Executes the pinned author TT-cross and backward smoother. Paper harmonizations
are explicit overlays. Run outputs are fresh, hashed and never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesfilter.testing.zhao_cui_reference_utils import (
    derive_full_sol_reference, derive_octave_class_overrides, derive_fixed_target_callbacks, extract_mlx_tree,
    sha256_file, tree_fingerprint, write_octave_shims, write_reference_logdensities,
)
SOURCE = ROOT / 'third_party/audit/zhao_cui_tensor_ssm_p10/source'
PLAN = 'docs/plans/zhao-cui-publication-replication-and-score-20261004.md'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Pinned source mismatch: {before[:100]!r}')
    return text.replace(before, after, 1)


def prepare(out, profile, *, pp_tail_precision_repair=False):
    derived = out / 'derived'
    records = dict(callbacks=extract_mlx_tree(SOURCE, derived),
                   octave_sampler=write_octave_shims(derived),
                   octave_classes=derive_octave_class_overrides(SOURCE, derived),
                   gaussian_logs=write_reference_logdensities(derived))
    records['full_solver'] = derive_full_sol_reference(
        SOURCE / 'models/full_sol.m', derived / 'models/full_sol_reference.m')
    pre = (SOURCE / 'models/pre_sol.m').read_text()
    pre = replace_once(pre, 'classdef pre_sol < full_sol',
                       'classdef pre_sol_reference < full_sol_reference')
    pre = replace_once(pre, 'function sol = pre_sol(', 'function sol = pre_sol_reference(')
    pre = replace_once(pre, 'sol@full_sol(', 'sol@full_sol_reference(')
    start = pre.index('        function [thetas, sams, w, varargout] = smooth(')
    end = pre.index('        function ESSs = smooth_t(', start)
    smooth = pre[start:end]
    smooth = replace_once(smooth,
        'logpdf_t = log(priorpdf(sol.model, fulldata(1:d+m, :, 1)));',
        'logpdf_t = reference_logprior(sol.model, fulldata(1:d+m, :, 1));')
    smooth = replace_once(smooth, 'log(transition(sol.model, fulldata(:, :, k+1), k))',
                          'reference_logtransition(sol.model, fulldata(:, :, k+1), k)')
    smooth = replace_once(smooth, 'log(like(sol.model, fulldata(:, :, k+1), k))',
                          'reference_loglike(sol.model, fulldata(:, :, k+1), k)')
    smooth = replace_once(smooth, 'w = logpdf_t - logpdf_e;', '''raw_log_weight = logpdf_t - logpdf_e;
            if any(isnan(raw_log_weight) | raw_log_weight==Inf) || ~any(isfinite(raw_log_weight))
                save('-mat7-binary',sprintf('invalid-smoothing-t%02d.mat',T), ...
                    'sams','thetas','logpdf_eall','logpdf_e','logpdf_t','raw_log_weight','fulldata');
                error('nonfinite smoothing weights; raw paths and density terms saved');
            end
            w = raw_log_weight;''')
    smooth = replace_once(smooth, 'if nargout == 4', 'if nargout >= 4')
    anchor = '                varargout{1} = logpdf_eall;\n            end'
    smooth = replace_once(smooth, anchor, anchor + '''
            if nargout >= 5
                % Source smooth subtracts sample minima from negative log q.
                % This constant cancels normalized weights, NOT absolute evidence.
                varargout{2} = NaN;
            end
            if nargout >= 6
                varargout{3} = struct('raw_log_weight',raw_log_weight, ...
                    'importance_ess',1/sum(w.^2),'finite_fraction',mean(isfinite(raw_log_weight)), ...
                    'absolute_evidence_valid',false);
            end''')
    pre = pre[:start] + smooth + pre[end:]
    (derived / 'models/pre_sol_reference.m').write_text(pre)
    records['pre_solver'] = dict(source='models/pre_sol.m:270-343',
        classification='extension_or_invention_observability_and_stable_gaussian_logs',
        source_sha256=sha256_file(SOURCE / 'models/pre_sol.m'))
    for cls in ('full_sol_reference','pre_sol_reference'):
        path = derived / 'models' / (cls + '.m')
        code = path.read_text()
        # Keep the author solve loop and calls; add observability only.
        anchor = "sol.weight(:, t) = w';"
        if anchor not in code:
            anchor = 'sol.weight(:, t) = w;'
        code = replace_once(code, anchor, anchor + '\n                reference_step_diagnostic(sol,t);')
        path.write_text(code)
    (derived / 'models/reference_step_diagnostic.m').write_text('''function reference_step_diagnostic(sol,t)
p = getenv('BAYESFILTER_REFERENCE_PROGRESS');
fid=fopen(p,'a');
rs=rank(sol.SIRTs{t}.approx);
fprintf(fid,'%d,%.17g,%.17g,%d\\n',t,sol.ESS_all(t),sol.FTT_time(t),max(rs));
fclose(fid);
if strcmp(getenv('BAYESFILTER_INCREMENTAL_SMOOTHING'),'1')
    terminals=str2num(getenv('BAYESFILTER_SMOOTH_TIMES'));
    if any(terminals==t), reference_save_smoothing(sol,t); end
end
end
''')
    (derived / 'models/reference_save_smoothing.m').write_text('''function reference_save_smoothing(sol,terminal)
repetitions=str2double(getenv('BAYESFILTER_SMOOTH_REPETITIONS'));
times=str2num(getenv('BAYESFILTER_REPEATED_TIMES'));
if ~any(times==terminal), repetitions=1; end
for replicate=1:repetitions
 reference_save_smoothing_once(sol,terminal,replicate);
end
end

function reference_save_smoothing_once(sol,terminal,replicate)
% Observability overlay of full_sol.m:139-206 / pre_sol.m:270-343.
% Restore RNG so intermediate smoothing cannot change any later TT fit.
saved_rand_seed=rand('seed'); saved_randn_seed=randn('seed');
unwind_protect
 seed=str2double(getenv('BAYESFILTER_SMOOTH_SEED'))+terminal+1000000*(replicate-1);
 rng(seed);
 smoothing_timer=tic;
 [thetas,sams,w,proposal_history,lml,stats]=smooth(sol,str2double(getenv('BAYESFILTER_SMOOTH_N')),terminal);
 smoothing_seconds=toc(smoothing_timer);
 if any(~isfinite(w)) || abs(sum(w)-1)>1e-10, error('invalid weights'); end
 raw_log_weight=stats.raw_log_weight;
 % pre_sol.m:270-343 has no author lml return or absolute density constant.
 legacy_mean_log_weight_available=isfield(stats,'legacy_mean_log_weight');
 if legacy_mean_log_weight_available
  legacy_mean_log_weight=stats.legacy_mean_log_weight;
 else
  legacy_mean_log_weight=NaN;
 end
 m=sol.model.m;
 path_quantiles=zeros(m,terminal+1,3);
 for k=1:terminal+1
  for j=1:m
   path_quantiles(j,k,:)=quantile(sams(j,:,k),[.05,.5,.95]);
  end
 end
 if replicate==1
 filename=sprintf('smoothing-t%02d.mat',terminal);
 save('-mat7-binary',[filename '.partial'],'thetas','sams','w','raw_log_weight','proposal_history','path_quantiles','smoothing_seconds','lml','legacy_mean_log_weight','legacy_mean_log_weight_available');
 [ok,message]=movefile([filename '.partial'],filename);
 if ~ok, error(message); end
 fid=fopen('smoothing-summary.csv','a');
 fprintf(fid,'%d,%.17g,%.17g,%.17g,%.17g,%.17g\\n',terminal,1/sum(w.^2),1/sum(w.^2)/numel(w),max(w),stats.finite_fraction,smoothing_seconds); fclose(fid);
 else
 filename=sprintf('smoothing-t%02d-r%03d.mat',terminal,replicate);
 ess=1/sum(w.^2); max_weight=max(w); zero_weight_count=sum(w==0);
 save('-mat7-binary',filename,'path_quantiles','smoothing_seconds','lml','ess','max_weight','zero_weight_count','seed');
 end
 fid=fopen('smoothing-repetitions.csv','a');
 fprintf(fid,'%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d\\n',terminal,replicate,seed,1/sum(w.^2),1/sum(w.^2)/numel(w),max(w),stats.finite_fraction,smoothing_seconds,sum(w==0)); fclose(fid);
 fprintf('SMOOTH_DONE t=%d replicate=%d ESS=%.9g/%d\\n',terminal,replicate,1/sum(w.^2),numel(w));
unwind_protect_cleanup
 rand('seed',saved_rand_seed); randn('seed',saved_randn_seed);
end_unwind_protect
end
''')
    records['incremental_smoothing'] = dict(
        classification='extension_or_invention_observability_only',
        source_anchors=['models/full_sol.m:139-206','models/pre_sol.m:270-343'],
        rng_restored=True, fitter_modified=False)
    changes = []
    if profile == 'paper':
        for model, filename in [('pp','predator_step.m'),('sir_austria','sir_step.m')]:
            path = derived / 'models' / model / filename
            before = sha256_file(path)
            path.write_text(replace_once(path.read_text(), 'x + fp3*delta/2', 'x + fp3*delta'))
            changes.append(dict(path=str(path), before_sha256=before,
                                after_sha256=sha256_file(path), reason='paper standard RK4 fourth stage'))
        path = derived / 'models/sir_austria/st_process.m'
        before = sha256_file(path)
        path.write_text(replace_once(path.read_text(),
            'X_new(1:2:model.m, :) = max(X_new(1:2:model.m, :), 0);',
            '% Paper model uses additive Gaussian noise without clipping.'))
        changes.append(dict(path=str(path), before_sha256=before,
                            after_sha256=sha256_file(path), reason='paper additive Gaussian state noise without clipping'))
    if profile == 'current_target':
        records['fixed_target_callbacks'] = derive_fixed_target_callbacks(derived)
    # Valid zero importance weights are retained in the sample count.
    full = derived / 'models/full_sol_reference.m'
    full.write_text(replace_once(full.read_text(), 'if any(~finite_mask(:))',
        'if any(isnan(raw_log_weight(:)) | raw_log_weight(:)==Inf) || ~any(finite_mask(:))'))
    if pp_tail_precision_repair:
        records['pp_tail_precision_repair'] = install_pp_tail_reference(derived, profile)
    records['paper_harmonizations'] = changes
    records['derived_tree_sha256'] = tree_fingerprint(derived)
    dump(out / 'extraction-manifest.json', records)
    return derived



def install_pp_tail_reference(derived, profile):
    """Copy an isolated reference helper; never edit the pinned author source."""
    helper = ROOT / 'bayesfilter/testing/zhao_cui_pp_tail_reference.py'
    copied = derived / 'reference_pp_tail.py'
    copied.write_bytes(helper.read_bytes())
    path = derived / 'models/reference_logtransition.m'
    code = path.read_text()
    k4 = '.5' if profile == 'author_driver' else '1'
    anchor = "value = -0.5*(model.m*log(2*pi) + 2*model.m*log(model.pre.sigma1) + sum(residual.^2,1));"
    added = r"""
% Arithmetic fallback only; every healthy floating-point value is unchanged.
bad = find(isnan(value) | value==Inf);
if ~isempty(bad) && strcmp(model.name,'pp')
    persistent reference_tail_call = 0;
    reference_tail_call = reference_tail_call + 1;
    folder=sprintf('tail-reference-%05d',reference_tail_call);
    if exist(folder,'dir'), error('tail evidence directory exists'); end
    mkdir(folder);
    input_path=fullfile(pwd,folder,'input.csv');
    output_path=fullfile(pwd,folder,'log-density.csv');
    record_path=fullfile(pwd,folder,'record.json');
    payload=[bad;theta(:,bad);previous(:,bad);thetax(model.d+1:model.d+model.m,bad)]';
    dlmwrite(input_path,payload,'precision',17);
    command=sprintf('"%s" "%s" --input "%s" --output "%s" --record "%s" --k4-fraction K4VALUE --sigma %.17g --dt %.17g', ...
        getenv('BAYESFILTER_REFERENCE_PYTHON'),'HELPERPATH',input_path,output_path,record_path,model.pre.sigma1,model.pre.dt);
    [status,message]=system(command);
    if status~=0, error(['high-precision reference failed: ' message]); end
    recovered=dlmread(output_path)';
    if numel(recovered)~=numel(bad) || any(isnan(recovered) | recovered==Inf)
        error('invalid recovered log density');
    end
    value(bad)=recovered;
end
""".replace('K4VALUE', k4).replace('HELPERPATH', str(copied).replace("'", "''"))
    path.write_text(replace_once(code, anchor, anchor+added))
    return dict(classification='extension_or_invention_independent_reference_arithmetic_adapter',
                helper_sha256=sha256_file(copied), precisions=[100,200],
                source_anchors=['paper Eq38 and p41', 'models/pp/predator_step.m', 'models/pp/odefun.m'],
                healthy_values_unchanged=True, k4_fraction=float(k4))


def make_script(args, out, derived):
    pp = args.model == 'pp'
    d,m,n = (6,2,2) if pp else (0,18,9)
    if args.profile == 'current_target': d = 0
    fit_n = args.fit_particles or (10000 if pp else 5000)
    als = args.als or (5 if args.profile != 'author_driver' else (4 if pp else 8))
    lowals = als if args.profile != 'author_driver' or args.als else 2
    initrank,kick = (min(10,args.rank),min(10,args.rank)) if pp else (min(20,args.rank),5)
    domain = "AlgebraicMapping(1)" if args.profile == 'current_target' or not pp or (args.profile == 'paper' and args.route == 'nonlinear') else "BoundedDomain([-1,1])"
    setup = 'myModel=complete(myModel);'
    target_parity = ''
    if args.profile == 'paper':
        if pp:
            setup = '''myModel.pre.ncons=[.1;.1;0;0;1;0];
myModel.pre.theta=[.6;.3;.5;.5;1.2;.5];
myModel.theta=norminv(myModel.pre.theta-myModel.pre.ncons);
initial=myModel.pre.init;'''
        else:
            setup = '''myModel.pre.priormean(1:2:m)=485+(1:9);
myModel.pre.priormean(2:2:m)=15-(1:9);
initial=myModel.pre.priormean;'''
        setup += '''
myModel.X=zeros(m,T+1); myModel.X(:,1)=initial; myModel.Y=zeros(n,T);
for t=1:T
 myModel.X(:,t+1)=st_process(myModel,[myModel.theta;myModel.X(:,t)],t);
 myModel.Y(:,t)=ob_process(myModel,[myModel.theta;myModel.X(:,t+1)],t);
end'''
    if args.profile == 'current_target':
        setup = f"myModel.Y=dlmread({quote(out / (args.model + '-input-observations.csv'))})'; myModel.X=[];"
        if pp:
            setup += "\nmyModel.theta=[]; myModel.pre.theta=[.6;.3;.5;.5;1.2;.5];"
        target_parity = f"""
previous=dlmread({quote(out / (args.model + '-probe-previous.csv'))})';
current=dlmread({quote(out / (args.model + '-probe-current.csv'))})';
if strcmp(name,'pp')
 predicted=predator_step(myModel,previous,repmat(myModel.pre.theta,1,size(previous,2)),'RK4');
else
 predicted=sir_step(previous,myModel.pre.theta);
end
pair=[current;previous];
actual=[predicted;reference_logprior(myModel,previous);reference_logtransition(myModel,pair,1);reference_loglike(myModel,pair,1)]';
expected=dlmread({quote(out / (args.model + '-probe-expected.csv'))});
errors=max(abs(actual-expected),[],1);
dlmwrite('target-parity-errors.csv',errors,'precision',17);
if any(~isfinite(errors)) || any(errors>1e-8), error('fixed target parity failed'); end
"""
    solver = f'sol=full_sol_reference(myModel,1,poly,opt,lowopt,N,{5 if pp else 4});'
    if args.route == 'nonlinear':
        if args.profile == 'paper':
            ref = '''precond.R=@(x) normcdf(x); precond.Rinv=@(x) norminv(x);
precond.r=@(x) exp(sum(-.5*x.^2-.5*log(2*pi),1));
precond.logr=@(x) sum(-.5*x.^2-.5*log(2*pi),1);'''
        else:
            ref = '''precond.R=@(x) tg_cdf(3*x,3); precond.Rinv=@(x) tg_inv(x,3)/3;
precond.r=@(x) 3*tg_pdf(3*x,3); precond.logr=@(x) log(3)+tg_logpdf(3*x,3);'''
        solver = ref + '\nprecond.c=.4; precond.opt="pifg";\nsol=pre_sol_reference(myModel,poly,opt,lowopt,N,5,precond);'
    times = getattr(args, 'save_times', None) or ([args.horizon] if args.terminal_only else list(range(1,args.horizon+1)))
    return f'''% Explicit CPU original-author-algorithm replication diagnostic.
cd({quote(SOURCE / 'deep-tensor.dev')}); load_dir;
cd({quote(out)});
addpath({quote(SOURCE / 'octave_compat')});
addpath({quote(SOURCE / 'models')}); addpath({quote(SOURCE / 'models/tensordot')});
addpath({quote(derived / 'models' / args.model)}); addpath({quote(derived / 'models')});
addpath(genpath({quote(derived / 'octave_overrides')}));
setenv('BAYESFILTER_REFERENCE_PROGRESS',{quote(out / 'fit-progress.csv')});
setenv('BAYESFILTER_INCREMENTAL_SMOOTHING','{0 if args.end_only_smoothing else 1}');
setenv('BAYESFILTER_SMOOTH_TIMES','{' '.join(map(str,times))}');
setenv('BAYESFILTER_SMOOTH_SEED','{args.smooth_seed}');
setenv('BAYESFILTER_SMOOTH_N','{args.smooth_samples}');
setenv('BAYESFILTER_SMOOTH_REPETITIONS','{args.smooth_repetitions}');
setenv('BAYESFILTER_REPEATED_TIMES','{' '.join(map(str,args.repeat_times or times))}');
fid=fopen('smoothing-summary.csv','w');
fprintf(fid,'time,ess,ess_fraction,max_weight,finite_fraction,seconds\\n'); fclose(fid);
fid=fopen('smoothing-repetitions.csv','w');
fprintf(fid,'time,replicate,seed,ess,ess_fraction,max_weight,finite_fraction,seconds,zero_weight_count\\n'); fclose(fid);
name='{args.model}'; d={d}; m={m}; n={n}; T={args.horizon}; N={fit_n};
rng(1); myModel=setup(ssmodel(name,d,m,n,T));
{setup}
{target_parity}
observations=myModel.Y; true_states=myModel.X; true_theta=myModel.theta;
if any(~isfinite(observations(:))) || any(~isfinite(true_states(:))), error('invalid data'); end
dlmwrite('observations.csv',observations','precision',17);
dlmwrite('true-states.csv',true_states','precision',17);
save('-mat7-binary','data.mat','observations','true_states','true_theta');
probe=priorsam(myModel,16); next=st_process(myModel,probe,1);
pair=[probe(1:d,:);next;probe(d+1:end,:)];
pdfs=[priorpdf(myModel,probe);transition(myModel,pair,1);like(myModel,pair,1)];
logs=[reference_logprior(myModel,probe);reference_logtransition(myModel,pair,1);reference_loglike(myModel,pair,1)];
mask=pdfs>realmin; log_error=max(abs(log(pdfs(mask))-logs(mask)));
if ~all(any(mask,2)) || ~isfinite(log_error) || log_error>1e-8, error('Gaussian log equivalence failed'); end
dlmwrite('gaussian-log-error.csv',log_error,'precision',17);
fid=fopen('call-chain.tsv','w');
for callback={{'full_sol_reference','pre_sol_reference','TTSIRT','transition','priorpdf','sir_step','predator_step'}}
 resolved=which(callback{{1}});
 if strcmp(resolved,'built-in function')
  if strcmp(callback{{1}},'TTSIRT'), resolved=file_in_loadpath('@TTSIRT/TTSIRT.m');
  else, resolved=file_in_loadpath([callback{{1}} '.m']); end
 end
 fprintf(fid,'%s\\t%s\\n',callback{{1}},resolved);
end
fclose(fid);
if ~strcmp(which('transition'),{quote(derived / 'models' / args.model / 'transition.m')}), error('wrong transition'); end
if {1 if args.prepare_only else 0}, fprintf('TARGET_PREPARATION_DONE\\n'); return; end
poly=ApproxBases(Lagrangep(4,8),{domain},d+2*m);
opt=TTOption('tt_method','random','als_tol',1e-10,'local_tol',1e-4,'max_rank',{args.rank},'max_als',{als},'init_rank',{initrank},'kick_rank',{kick});
lowopt=TTOption('tt_method','random','als_tol',1e-10,'local_tol',1e-4,'max_rank',{args.rank},'max_als',{lowals},'init_rank',{initrank},'kick_rank',{kick});
rng({args.fit_seed});
{solver}
fit_timer=tic; sol=solve(sol); fit_seconds=toc(fit_timer);
fit_ess=sol.ESS_all; fit_times=sol.FTT_time;
save('-mat7-binary','fit-diagnostics.mat','fit_ess','fit_times','fit_seconds');
if strcmp(getenv('BAYESFILTER_INCREMENTAL_SMOOTHING'),'0')
 for terminal=[{' '.join(map(str,times))}]
  reference_save_smoothing(sol,terminal);
 end
end
fprintf('PUBLICATION_REPLICATION_DONE\\n');
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--model',choices=['pp','sir_austria'],required=True)
    p.add_argument('--route',choices=['linear','nonlinear'],default='linear')
    p.add_argument('--profile',choices=['paper','author_driver','current_target'],default='paper')
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--pp-tail-precision-repair',action='store_true',help='Independent high-precision replay of exceptional PP transition densities')
    p.add_argument('--reference-python',default='/home/chakwong/anaconda3/envs/tftwogpu/bin/python')
    p.add_argument('--prepare-only',action='store_true',help='Run target checks only; no TT fitting or smoothing')
    p.add_argument('--plan-file',default=PLAN)
    p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--save-times',type=int,nargs='+',help='explicit incremental smoothing checkpoints; fitter RNG is restored')
    p.add_argument('--fit-particles',type=int)
    p.add_argument('--smooth-samples',type=int,default=10000)
    p.add_argument('--smooth-repetitions',type=int,default=1)
    p.add_argument('--repeat-times',type=int,nargs='*',help='Default repeats at every saved time; explicit times narrow the reporting scope')
    p.add_argument('--als',type=int)
    p.add_argument('--fit-seed',type=int,default=2)
    p.add_argument('--smooth-seed',type=int,default=3000)
    p.add_argument('--timeout-seconds',type=int,default=64800)
    p.add_argument('--terminal-only',action='store_true')
    p.add_argument('--end-only-smoothing',action='store_true',help='Diagnostic parity mode; default saves each completed update')
    args=p.parse_args()
    horizon_cap = 50 if args.profile == 'current_target' else 20
    timeout_cap = 129600 if args.profile == 'current_target' else 64800
    if not 1<=args.horizon<=horizon_cap or not 2<=args.rank<=40 or not 1<=args.timeout_seconds<=timeout_cap:
        p.error('outside planned bounds')
    if not 1<=args.smooth_repetitions<=40: p.error('between 1 and 40 smoothing repetitions')
    if args.save_times and (args.terminal_only or len(set(args.save_times)) != len(args.save_times) or any(t < 1 or t > args.horizon for t in args.save_times)):
        p.error('invalid or conflicting smoothing checkpoint times')
    if args.repeat_times and any(t<1 or t>args.horizon for t in args.repeat_times): p.error('invalid repetition times')
    if args.pp_tail_precision_repair and args.model!='pp': p.error('PP arithmetic adapter only')
    if args.profile=='current_target' and (args.dataset is None or args.route!='linear'):
        p.error('current target requires exact dataset and linear conditional solver')
    if args.profile!='current_target' and args.dataset is not None: p.error('dataset applies only to current_target')
    if args.route=='nonlinear' and args.model!='pp': p.error('unplanned nonlinear model')
    if args.fit_particles is not None and (args.fit_particles<64 or args.fit_particles%2): p.error('even fit cloud >=64 required')
    if args.smooth_samples<64 or (args.als is not None and args.als<1): p.error('invalid smoke configuration')
    out=args.output_root.resolve()
    if out==SOURCE or SOURCE in out.parents: p.error('immutable source')
    out.mkdir(parents=True,exist_ok=False)
    (out/'runner-source.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic()
    fingerprint=tree_fingerprint(SOURCE)
    manifest=dict(schema='zhao_cui_publication_replication.v2',plan=args.plan_file,
      result_file=str(out/'result.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      command=[sys.executable,*sys.argv],environment=sys.executable,
      octave_version=subprocess.check_output(['octave-cli','--version'],text=True).splitlines()[0],
      settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
      cpu_only=True,gpu_status='intentionally hidden with CUDA_VISIBLE_DEVICES=-1',
      threads=2,started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      source_tree_sha256_before=fingerprint,runner_sha256=sha256_file(Path(__file__)),
      status='preparing',exact_matlab_dataset_replication=False)
    dump(out/'manifest.json',manifest)
    try:
        if args.profile=='current_target':
            from types import SimpleNamespace
            from run_zhao_cui_reference import prepare_fixed_data
            os.environ['TF_NUM_INTRAOP_THREADS']='2'
            os.environ['TF_NUM_INTEROP_THREADS']='1'
            target_records=prepare_fixed_data([args.dataset],out,SimpleNamespace(models=[args.model],horizon=args.horizon))
            dump(out/'target-preparation.json',target_records)
        derived=prepare(out,args.profile,pp_tail_precision_repair=args.pp_tail_precision_repair)
        script=out/'run.m'; script.write_text(make_script(args,out,derived))
        manifest['script_sha256']=sha256_file(script)
        manifest['status']='running'; dump(out/'manifest.json',manifest)
        env=dict(os.environ,BAYESFILTER_REFERENCE_PYTHON=args.reference_python,CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
            proc=subprocess.Popen(['octave-cli','--quiet','--no-gui',str(script)],cwd=out,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            manifest['octave_pid']=proc.pid; dump(out/'manifest.json',manifest)
            try:
                code=proc.wait(timeout=args.timeout_seconds)
                marker='TARGET_PREPARATION_DONE' if args.prepare_only else 'PUBLICATION_REPLICATION_DONE'
                status=('prepared' if args.prepare_only else 'complete') if code==0 and marker in (out/'stdout.log').read_text() else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                code=proc.returncode; status='timeout'
        result=dict(status=status,exit_code=code)
        summary=out/'smoothing-summary.csv'
        if summary.exists():
            with summary.open() as stream: result['smoothing']=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        result['data_sha256']={name:sha256_file(out/name) for name in ['observations.csv','true-states.csv'] if (out/name).exists()}
    except Exception as exc:
        result=dict(status='preparation_failed',failure_type=type(exc).__name__,reason=str(exc))
    result['wall_seconds']=time.monotonic()-started
    result['source_tree_unchanged']=tree_fingerprint(SOURCE)==fingerprint
    if not result['source_tree_unchanged']: result['status']='invalid_source_mutation'
    dump(out/'result.json',result)
    manifest.update(status=result['status'],wall_seconds=result['wall_seconds'],source_tree_unchanged=result['source_tree_unchanged'])
    dump(out/'manifest.json',manifest)
    print(json.dumps(dict(output=str(out),**result),allow_nan=False))
    return 0 if result['status'] in ('complete','prepared') else 1


if __name__=='__main__': raise SystemExit(main())
