#!/usr/bin/env python3
"""Audit/reference-only controller for the pinned Zhao--Cui Octave source.

Uses the author full_sol call chain; it never imports source code into a
production filtering lane. Reduced settings are mechanics checks, not accuracy
or score-oracle evidence. All products are written to a fresh result directory.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesfilter.testing.zhao_cui_reference_utils import (
    EXTRACTOR_VERSION, derive_full_sol_reference, extract_mlx_tree, derive_octave_class_overrides,
    sha256_file, summarize_log_weights, write_octave_shims, write_reference_logdensities,
    derive_fixed_target_callbacks, tree_fingerprint,
)

PLAN = "docs/plans/zhao-cui-reference-repair-20261003.md"
AUDIT = ROOT / "third_party/audit/zhao_cui_tensor_ssm_p10"
SOURCE = AUDIT / "source"
COMMIT = "80034dccb99eb1d86284a1839b4a12067d13b9da"


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def make_case(model, derived, out, horizon, particles, rank, fixed=False):
    d, m, n = (6, 2, 2) if model == "pp" else (0, 18, 9)
    if fixed:
        d = 0
    domain = "BoundedDomain([-1, 1])" if model == "pp" and not fixed else "AlgebraicMapping(1)"
    epd = 5 if model == "pp" else 4
    setup = "myModel = complete(myModel);"
    parity = ""
    if fixed:
        setup = f"myModel.Y = dlmread({quote(out / (model + '-input-observations.csv'))})';"
        if model == "pp":
            setup += "\nmyModel.theta = []; myModel.pre.theta = [0.6;0.3;0.5;0.5;1.2;0.5];"
        parity = f"""
previous = dlmread({quote(out / (model + '-probe-previous.csv'))})';
current = dlmread({quote(out / (model + '-probe-current.csv'))})';
if strcmp(name,'pp')
    predicted = predator_step(myModel,previous,repmat(myModel.pre.theta,1,size(previous,2)),'RK4');
else
    predicted = sir_step(previous,myModel.pre.theta);
end
pair = [current; previous];
actual = [predicted; reference_logprior(myModel,previous); ...
    reference_logtransition(myModel,pair,1); reference_loglike(myModel,pair,1)]';
expected = dlmread({quote(out / (model + '-probe-expected.csv'))});
dlmwrite({quote(out / (model + '-probe-actual.csv'))},actual,'precision',17);
errors = max(abs(actual-expected),[],1);
dlmwrite({quote(out / (model + '-parity-errors.csv'))},errors,'precision',17);
if any(~isfinite(errors)) || any(errors > 1e-8), error('fixed target parity failed'); end
"""
    return f"""% Derived audit runner; reduced settings; pinned source is immutable.
cd({quote(SOURCE / 'deep-tensor.dev')});
load_dir;
cd({quote(out)});
addpath({quote(SOURCE / 'octave_compat')});
addpath({quote(SOURCE / 'models')});
addpath({quote(SOURCE / 'models/tensordot')});
addpath({quote(SOURCE / 'models' / model)});
addpath({quote(derived / 'models' / model)});
addpath({quote(derived / 'models')});
addpath(genpath({quote(derived / 'octave_overrides')}));
name = '{model}'; d = {d}; m = {m}; n = {n};
T = {horizon}; N = {particles};
rng(1);
myModel = setup(ssmodel(name, d, m, n, T));
{setup}
observations = myModel.Y;
truth = myModel.theta;
dlmwrite({quote(out / (model + '-observations.csv'))}, observations', 'precision', 17);
if any(~isfinite(observations(:))), error('nonfinite source data'); end
% Check the stable log evaluation wherever the inspected PDF is positive.
probe = priorsam(myModel,8);
next_probe = st_process(myModel,probe,1);
pair_probe = [probe(1:d,:); next_probe; probe(d+1:end,:)];
pdfs = [priorpdf(myModel,probe); transition(myModel,pair_probe,1); like(myModel,pair_probe,1)];
logs = [reference_logprior(myModel,probe); reference_logtransition(myModel,pair_probe,1); reference_loglike(myModel,pair_probe,1)];
mask = pdfs > realmin;
if ~all(any(mask,2)), error('no healthy Gaussian equivalence probes'); end
log_error = max(abs(log(pdfs(mask))-logs(mask)));
if ~isfinite(log_error) || log_error > 1e-8, error('stable Gaussian log mismatch'); end
dlmwrite({quote(out / (model + '-gaussian-log-error.csv'))},log_error,'precision',17);
{parity}
fid = fopen({quote(out / (model + '-call-chain.tsv'))},'w');
for callback = {{'full_sol_reference','TTSIRT','transition','priorpdf','st_process','reference_logtransition'}}
    resolved = which(callback{{1}});
    if strcmp(resolved,'built-in function')
        if strcmp(callback{{1}},'TTSIRT'), resolved = file_in_loadpath('@TTSIRT/TTSIRT.m');
        else, resolved = file_in_loadpath([callback{{1}} '.m']); end
    end
    fprintf(fid,'%s\\t%s\\n',callback{{1}},resolved);
end
fclose(fid);
if ~strcmp(which('transition'),{quote(derived / 'models' / model / 'transition.m')}), error('wrong transition callback'); end
if ~strcmp(file_in_loadpath('full_sol_reference.m'),{quote(derived / 'models/full_sol_reference.m')}), error('wrong solver class'); end
poly = ApproxBases(Lagrangep(4, 8), {domain}, d + 2*m);
opt = TTOption('tt_method', 'random', 'als_tol', 1E-10, ...
    'local_tol', 1E-4, 'max_rank', {rank}, 'max_als', 1, ...
    'init_rank', 2, 'kick_rank', 1);
rng(2);
sol = full_sol_reference(myModel, 1, poly, opt, opt, N, {epd});
if ~isa(sol,'full_sol_reference'), error('wrong solver instance'); end
sol = solve(sol);
rng(3);
[thetas, sams, w, proposal_history, lml, stats] = smooth(sol, N, T);
raw_weights = stats.raw_log_weight;
dlmwrite({quote(out / (model + '-raw-log-weights.csv'))}, raw_weights(:), 'precision', 17);
save('-mat7-binary', {quote(out / (model + '.mat'))}, ...
    'observations', 'truth', 'thetas', 'sams', 'w', 'proposal_history', 'lml', 'stats');
fid = fopen({quote(out / (model + '-summary.tsv'))}, 'w');
fprintf(fid, 'corrected_logmeanexp\\t%.17g\\n', lml);
fprintf(fid, 'legacy_mean_log_weight\\t%.17g\\n', stats.legacy_mean_log_weight);
fprintf(fid, 'finite_fraction\\t%.17g\\n', stats.finite_fraction);
fprintf(fid, 'importance_ess\\t%.17g\\n', stats.importance_ess);
fprintf(fid, 'corrected_status\\t%d\\n', stats.corrected_status);
fprintf(fid, 'tt_normalizer_accumulator\\t%.17g\\n', sol.logmarginal_likelihood);
fclose(fid);
if stats.corrected_status ~= 1, error('invalid corrected log weights'); end
fprintf('ZHAO_CUI_REFERENCE_DONE model=%s\\n', name);
"""


def run_case(model, derived, out, args):
    script = out / (model + ".m")
    script.write_text(make_case(model, derived, out, args.horizon, args.particles, args.rank, args.mode == "fixed_target"))
    command = ["octave-cli", "--quiet", "--no-gui", str(script)]
    started = time.monotonic()
    timed_out = False
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    with (out / (model + ".stdout.log")).open("w") as stdout, \
         (out / (model + ".stderr.log")).open("w") as stderr:
        process = subprocess.Popen(command, cwd=out, env=environment, stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    values = {}
    summary = out / (model + "-summary.tsv")
    if summary.exists():
        for line in summary.read_text().splitlines():
            key, value = line.split("\t")
            number = float(value)
            values[key] = number if math.isfinite(number) else None
    raw_path = out / (model + "-raw-log-weights.csv")
    parity = None
    if raw_path.exists():
        raw = [float(row[0]) for row in csv.reader(raw_path.open())]
        checked = summarize_log_weights(raw)
        computed = values.get("corrected_logmeanexp")
        parity = (checked.valid and computed is not None
                  and math.isclose(checked.corrected_logmeanexp, computed,
                                   rel_tol=1e-12, abs_tol=1e-12))
    passed = code == 0 and values.get("corrected_status") == 1 and parity is True
    stderr_text = (out / (model + ".stderr.log")).read_text()
    failure = None
    if not passed:
        failure = ("budget_timeout" if timed_out else "octave_compatibility"
                   if any(x in stderr_text for x in ("undefined", "parse error", "invalid call"))
                   else "numerical_or_source_route")
    parity_file = out / (model + "-parity-errors.csv")
    target_parity = None
    if parity_file.exists():
        errors = [float(x) for x in parity_file.read_text().strip().split(",")]
        target_parity = dict(max_absolute_error=max(errors),
                             passed=all(math.isfinite(x) and x <= 1e-8 for x in errors),
                             components="transition means, initial/transition/observation log densities",
                             states=5, tolerance=1e-8)
    gaussian_file = out / (model + "-gaussian-log-error.csv")
    return dict(model=model, command=command, mode=args.mode,
                target_parity=target_parity,
                stable_log_pdf_max_error=float(gaussian_file.read_text()) if gaussian_file.exists() else None,
                status="passed" if passed else "failed",
                returncode=code, failure_class=failure,
                wall_seconds=time.monotonic()-started,
                independent_summary_parity=parity, values=values,
                scope=dict(horizon=args.horizon, particles=args.particles,
                           rank=args.rank, als_passes=1),
                evidence_role="mechanics_only", analytical_score=None)


def alignment_reports(paths):
    """Report known source mismatches; never manufacture an aligned claim."""
    reports = []
    for path in paths:
        payload, raw, origin = load_dataset(path)
        model = payload.get("model") or {"canonical_predator_prey_x0_then_transition_observe_v1": "predator_prey", "canonical_sir_d18_x0_then_transition_observe_v1": "sir_d18"}.get(payload.get("target_id"))
        reasons = []
        if model == "predator_prey":
            reasons = [
                "Author pp driver integrates six parameters; current likelihood conditions on six physical parameters.",
                "Author ODE maps K=90+20*theta(5) and a=20+10*theta(6); its true physical vector differs from current [0.6,114,25,0.3,0.5,0.5].",
                "Author predator_step uses a half-step in the fourth RK stage; current standard RK4 uses the full step.",
            ]
        elif model == "sir_d18":
            reasons = [
                "At zero log scales the source SIR callbacks match the current half-step RK convention; executable parity is still required.",
                "Source-generated observations are a different dataset; saved observations must be injected before comparison.",
                "Parameter derivatives are absent from the original code.",
            ]
        else:
            reasons = ["Dataset model could not be identified; alignment was not checked."]
        reports.append(dict(
            dataset=origin, dataset_sha256=hashlib.sha256(raw).hexdigest(),
            observation_sha256=payload.get("observation_sha256"),
            target_id=payload.get("target_id"), model=model, horizon=len(payload.get("observations", [])),
            status="not_aligned" if model in ("predator_prey", "sir_d18") else "not_checked",
            reasons=reasons, reference_admission="blocked",
            next_step="Explicit fixed-target adapter plus executable density and transition parity.",
        ))
    return reports or [dict(status="not_checked", reference_admission="blocked",
                            reason="No current saved dataset was supplied.")]


def load_dataset(path):
    path = path.resolve()
    if path.exists():
        data = path.read_bytes()
        origin = str(path)
    else:
        relative = path.relative_to(ROOT).as_posix()
        data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        origin = git("rev-parse", "HEAD") + ":" + relative
    return json.loads(data), data, origin


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def prepare_fixed_data(paths, out, args):
    # Tiny eager CPU reference probes, not a training/production execution lane.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.highdim.ledh_canonical_score_stages_tf import _gaussian_log_density_and_tangent

    models = {"pp": "predator_prey", "sir_austria": "sir_d18"}
    loaded = {d[0]["target_id"]: d for d in (load_dataset(path) for path in paths)}
    records = []
    for source_model in args.models:
        spec = NonlinearSQMCSpec(models[source_model])
        if spec.target_id not in loaded:
            raise ValueError(f"missing exact dataset for {spec.target_id}")
        payload, raw, origin = loaded[spec.target_id]
        observations = payload["observations"]
        if len(observations) != args.horizon or any(len(row) != spec.observation_dimension for row in observations):
            raise ValueError("dataset shape/horizon mismatch; no implicit slicing")
        tensor = tf.convert_to_tensor(observations, tf.as_dtype(payload["dtype"]))
        actual_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()
        if actual_hash != payload["observation_sha256"]:
            raise ValueError("saved observation hash mismatch")
        (out / (source_model + "-input-dataset.json")).write_bytes(raw)
        write_csv(out / (source_model + "-input-observations.csv"), observations)
        theta = spec.default_theta(tf.float64)
        model, _ = spec.model(theta, tf.zeros_like(theta))
        mean = spec.initial_mean(tf.float64)
        base = mean.numpy().tolist()
        previous = tf.constant([base, [x+1 for x in base], [x-1 for x in base],
                                [0.01,0.1]*(spec.dimension//2),
                                [50.0,20.0]*(spec.dimension//2)], tf.float64)
        prediction = model.transition_mean_fn(theta, previous)
        current = prediction + tf.constant([0.2,-0.2]*(spec.dimension//2), tf.float64)
        initial_log = -0.5*(spec.dimension*math.log(2*math.pi)+tf.reduce_sum((previous-mean)**2,axis=1))
        covariance = model.process_covariance
        residual = current-prediction
        transition_log = -0.5*(spec.dimension*math.log(2*math.pi)+tf.linalg.logdet(covariance)
                               +tf.reduce_sum(residual*tf.transpose(tf.linalg.solve(covariance,tf.transpose(residual))),axis=1))
        observation = tf.constant(observations[0],tf.float64)
        if model.observation_log_density_fn is not None:
            observed_log = model.observation_log_density_fn(theta,current,observation)
        else:
            observed = model.observation_fn(current)
            observed_log, _ = _gaussian_log_density_and_tangent(
                tf.broadcast_to(observation[None,:],tf.shape(observed)),None,
                observed,None,tf.linalg.cholesky(model.observation_covariance))
        expected = tf.concat([prediction,initial_log[:,None],transition_log[:,None],observed_log[:,None]],axis=1)
        for name,value in (("previous",previous),("current",current),("expected",expected)):
            write_csv(out / (source_model + "-probe-" + name + ".csv"),value.numpy().tolist())
        records.append(dict(source_model=source_model, model=spec.name,target_id=spec.target_id,
                            dataset_origin=origin,dataset_sha256=hashlib.sha256(raw).hexdigest(),
                            observation_sha256=actual_hash,theta=theta.numpy().tolist(),
                            horizon=args.horizon,status="pending_executable_parity",
                            classification="extension_or_invention_fixed_target",
                            parity_environment="CPU eager float64 independent reference; GPU intentionally hidden",
                            parameter_provenance="NonlinearSQMCSpec.default_theta at recorded Git commit",
                            reference_admission="blocked_until_parity; accuracy_unchecked"))
    return records


def write_literature_ledger(out):
    dump(out / "literature-ledger.json", dict(
        seed_paper="Zhao and Cui, JMLR 2024, Tensor-Train Methods for Sequential State and Parameter Learning in State-Space Models",
        local_paper=".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf",
        upstream_commit=COMMIT,
        claims=[
            dict(claim="recursive two-slice TT filtering density",paper="Algorithm 1, equation (12)",source="models/full_sol.m:21-134",status="inspected"),
            dict(claim="author SIR and PP model definitions",paper="Sections 6.3 and 6.4",source="eg3_sir/mainscript.m; eg4_predatorprey/mainscript.m; extracted models/*/*.m",status="inspected; source RK half-step differs from textbook RK4"),
            dict(claim="legacy reported lml is mean log weight",paper="not a paper claim; local source finding",source="models/full_sol.m:181-205",status="checked"),
            dict(claim="corrected logmeanexp",paper="local importance-sampling derivation in plan",source="derived full_sol_reference.m",status="local correction, subject to proposal support and Monte Carlo error"),
            dict(claim="observed-data analytical score",paper="not established by inspected route",source="models/full_sol.m",status="unavailable in this source route"),
        ],
        snowballing=dict(backward="bounded to Algorithm 1 and source dependencies used by these examples",forward="not performed; no completeness or novelty claim"),
        omission_risk="No MATLAB parity, convergence study, or score-oracle certification. Octave overlays are local adaptations."))


def write_result_note(out, cases, manifest):
    lines = ["# Zhao--Cui source reference execution", "",
             "This is a bounded Octave mechanics check of the pinned author solver with recorded local adaptations.",
             "The corrected statistic averages importance weights before taking the logarithm. It remains sensitive to proposal support and Monte Carlo error.", "",
             "| Model | Status | Corrected logmeanexp | Legacy mean log weight | ESS | Finite fraction |",
             "|---|---|---:|---:|---:|---:|"]
    for case in cases:
        v=case.get("values",{})
        lines.append("| {} | {} | {} | {} | {} | {} |".format(case["model"],case["status"],v.get("corrected_logmeanexp"),v.get("legacy_mean_log_weight"),v.get("importance_ess"),v.get("finite_fraction")))
    lines += ["", "| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |",
              "|---|---|---|---|---|---|",
              "| Preserve source reference mechanics | See numerical status and parity in results.json | Nonfinite/provenance/alignment failures block numerical admission | Tiny rank and sample size; support and Monte Carlo error | Convergence and independent-reference validation before oracle use | No score, accuracy, ranking, production, or HMC certification |", "",
              "| Inference item | Status |", "|---|---|",
              "| Hard veto screen | Per-case finite-weight and source-provenance checks in results.json |",
              "| Statistically supported ranking | None; no method comparison |",
              "| Descriptive differences | Corrected and legacy values, ESS, TT normalizer, runtime |",
              "| Default readiness | Not evaluated |",
              "| Next evidence | Adequate TT/sample convergence, uncertainty, support checks and independent likelihood comparison |", "",
              "Post-run red team: a finite corrected value can be very inaccurate when a single trajectory dominates. Model parity tests do not test TT posterior quality. The source PP bounded proposal may miss target mass. Multi-setting convergence with independent reference agreement could overturn these accuracy concerns.", "",
              "No analytical observed-data score is implemented by this bridge. Raw log weights, path samples, resolved call chain and exact commands are retained.", "",
              f"CPU wall seconds: {manifest['wall_seconds']:.3f}. Mode: {manifest['mode']}. Plan: {PLAN}."]
    (out / "result-note.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("author", "fixed_target"), default="author")
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--models", nargs="+", choices=("pp", "sir_austria"),
                        default=["pp", "sir_austria"])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.horizon <= 20 and args.particles >= 64 and args.particles % 2 == 0
            and 2 <= args.rank <= 40 and 1 <= args.timeout_seconds <= 450):
        parser.error("invalid bounded source-reproduction settings")
    out = args.output_root.resolve()
    if out == SOURCE or SOURCE in out.parents:
        parser.error("the pinned source cannot be an output directory")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    snapshot_before = tree_fingerprint(SOURCE)
    preflight = dict(schema="zhao_cui_reference_preflight.v1", plan=PLAN,
                     git_commit=git("rev-parse", "HEAD"), command=[sys.executable,*sys.argv],
                     environment=sys.executable, mode=args.mode, cpu_only=True,
                     gpu_status="intentionally hidden for Octave and TensorFlow probes",
                     source_tree_sha256=snapshot_before, status="preflight_started")
    dump(out / "preflight-manifest.json", preflight)
    try:
        prepared_data = prepare_fixed_data(args.dataset, out, args) if args.mode == "fixed_target" else []
    except Exception as exc:
        preflight.update(status="preflight_failed", failure_type=type(exc).__name__,
                         reason=str(exc), wall_seconds=time.monotonic()-started)
        dump(out / "preflight-manifest.json", preflight)
        raise
    derived = out / "derived"
    records = extract_mlx_tree(SOURCE, derived)
    shim_record = write_octave_shims(derived)
    cdf_record = derive_octave_class_overrides(SOURCE, derived)
    log_records = write_reference_logdensities(derived)
    fixed_records = derive_fixed_target_callbacks(derived) if args.mode == "fixed_target" else []
    class_record = derive_full_sol_reference(SOURCE / "models/full_sol.m",
                                             derived / "models/full_sol_reference.m")
    dump(out / "extraction-manifest.json", dict(
        schema="zhao_cui_source_extraction.v1", extractor=EXTRACTOR_VERSION,
        upstream_commit=COMMIT, compatibility_patch_sha256=sha256_file(AUDIT / "octave_compatibility.patch"),
        callbacks=records, corrected_class=class_record, octave_shims=shim_record, cdf_compatibility=cdf_record, logdensities=log_records, fixed_target_callbacks=fixed_records))
    alignment = prepared_data if args.mode == "fixed_target" else alignment_reports(args.dataset)
    dump(out / "alignment-report.json", alignment)
    write_literature_ledger(out)
    manifest = dict(
        schema="zhao_cui_author_reference.v2", plan=PLAN, result_file=str(out / "results.json"),
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        git_commit=git("rev-parse", "HEAD"), working_tree_status=git("status", "--short"),
        implementation_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in [Path(__file__).resolve(), ROOT / "bayesfilter/testing/zhao_cui_reference_utils.py"]},
        command=[sys.executable, *sys.argv], python=sys.executable,
        octave_version=subprocess.check_output(["octave-cli", "--version"], text=True).splitlines()[0],
        cpu_only=True, gpu_status="intentionally hidden: CUDA_VISIBLE_DEVICES=-1",
        environment=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        seeds=dict(data=1, tt=2, smoothing=3), source_commit=COMMIT,
        source_tree_sha256_before=snapshot_before,
        source_sha256_before=sha256_file(SOURCE / "models/full_sol.m"),
        cases=[], score_status="not_implemented_in_author_source",
        evidence_role="mechanics_only_no_accuracy_or_ranking_claim",
        mode=args.mode, data_version=prepared_data or "author-generated using extracted setup and complete; saved CSV per case")
    dump(out / "run-manifest.json", manifest)
    preflight.update(status="preflight_passed", wall_seconds=time.monotonic()-started)
    dump(out / "preflight-manifest.json", preflight)
    cases = []
    if not args.extract_only:
        for model in args.models:
            case = run_case(model, derived, out, args)
            cases.append(case)
            dump(out / "results.json", cases)
            print(json.dumps(case, allow_nan=False), flush=True)
    manifest.update(cases=cases, wall_seconds=time.monotonic()-started,
                    source_sha256_after=sha256_file(SOURCE / "models/full_sol.m"))
    manifest["source_tree_sha256_after"] = tree_fingerprint(SOURCE)
    if manifest["source_tree_sha256_after"] != snapshot_before:
        raise RuntimeError("pinned source tree changed")
    if args.mode == "fixed_target" and cases:
        for record in alignment:
            case = next(row for row in cases if row["model"] == record["source_model"])
            parity = case.get("target_parity") or {}
            record.update(status="aligned_at_tested_point" if parity.get("passed") else "not_aligned",
                          target_parity=parity, reference_admission="mechanics_only_accuracy_unchecked")
        dump(out / "alignment-report.json", alignment)
    if manifest["source_sha256_after"] != manifest["source_sha256_before"]:
        raise RuntimeError("pinned source changed")
    dump(out / "run-manifest.json", manifest)
    if not cases:
        dump(out / "results.json", [])
    write_result_note(out, cases, manifest)
    return 0 if all(case["status"] == "passed" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
