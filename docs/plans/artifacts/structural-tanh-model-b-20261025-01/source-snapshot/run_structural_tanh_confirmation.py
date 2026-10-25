#!/usr/bin/env python3
"""Fresh-data Model B confirmation, reusing the structural campaign workers."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys

import run_structural_ledh_ab as base

PLAN = 'docs/plans/structural-tanh-model-b-confirmation-2026-10-25.md'
SCOPES = [('model_b', 2), ('model_b', 50)]
SEED_OFFSET = 200000
RESET = 'conditional_fp64'


def nomination_loss(row, reference):
    centered = [(v-row['value'][0])-(r['mean'][0]-reference[0]['mean'][0])
                for v, r in zip(row['value'], reference)]
    errors = [[s-r for s, r in zip(score, ref['mean'][1:])]
              for score, ref in zip(row['score'], reference)]
    return (sum(x*x for x in centered)/len(centered) +
            sum(x*x/(1+r*r) for e, ref in zip(errors, reference)
                for x, r in zip(e, ref['mean'][1:]))/(len(errors)*len(errors[0])))


def compare_probe(row, reference):
    """Descriptive accuracy table; finite-PF bias is never called resolved."""
    result = []
    for i, (value, score, ref, heuristic) in enumerate(zip(
            row['value'], row['score'], reference['estimates'], reference['heuristics'])):
        estimate = [value, *score]
        error = [a-b for a, b in zip(estimate, ref['mean'])]
        envelope = [4*s+r for s, r in zip(ref['mcse'], ref['refinement'])]
        scales = [max(1., abs(ref['mean'][0]))] + [max(1., math.sqrt(
            sum(x*x for x in ref['mean'][1:])))]*len(score)
        heuristic_errors = {}
        for name in ('ekf', 'cubature', 'bootstrap'):
            arm = heuristic.get(name)
            if arm is not None:
                values = arm['mean'] if 'mean' in arm else [arm['value'], *arm['score']]
                heuristic_errors[name] = [a-b for a,b in zip(values, ref['mean'])]
        result.append(dict(region=heuristic['region'], q=ref['q'], ledh=estimate,
            reference=ref['mean'], reference_mcse=ref['mcse'],
            reference_refinement=ref['refinement'], signed_error=error,
            outside_uncertainty_plus_one_percent=[abs(e)>u+.01*s
                for e, u, s in zip(error, envelope, scales)],
            reference_kind=ref['kind'], heuristics=heuristic, heuristic_signed_errors=heuristic_errors,
            heuristic_dominance_verdict='UNRESOLVED_NO_REPLICATED_LEDH_RANKING',
            statistically_supported_ranking=False))
    return result


class Confirmation(base.Campaign):
    def __init__(self, root, hours=None):
        if hours is not None and not 0 < hours <= 12:
            raise ValueError('This confirmation is capped at 12 GPU-worker hours')
        super().__init__(root, hours)
        if self.state.get('confirmation_seed_offset', SEED_OFFSET) != SEED_OFFSET:
            raise ValueError('Fresh-data seed policy changed')
        self.state.update(plan=PLAN, scopes=SCOPES, confirmation_seed_offset=SEED_OFFSET,
                          reset_precision_policy=RESET)
        self.persist()
        manifest = self.root/'run-manifest.json'
        if not manifest.exists():
            base.save(manifest, dict(git_commit=self.state['git_commit'], branch=self.state['branch'],
                command=[sys.executable, *sys.argv], executable=sys.executable,
                source_identity=self.state['source_identity'], plan=PLAN,
                plan_sha256=base.hash_file(base.ROOT/PLAN), created=base.now(),
                gpu_budget_seconds=self.state['gpu_budget_seconds'],
                cpu_budget_seconds=self.state['cpu_budget_seconds'], device=base.GPU,
                data='New synthetic data; preparation receipts record exact seeds and hashes',
                seed_offset=SEED_OFFSET, training_seeds=[17, 29],
                worker_environments='attempts/*/environment.json',
                elapsed_and_failures='state.json: attempts', result=str(self.root/'analysis.json')))
            for name in ('run_structural_ledh_ab.py', 'run_structural_tanh_confirmation.py'):
                path=self.root/'source-snapshot'/name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes((base.ROOT/'scripts'/name).read_bytes())

    def stop(self, status, action):
        self.state['next_action'] = action
        return self.finish(status)

    def numerical_task(self, key, kind, allowance, **kwargs):
        return self.task(key, kind, allowance, model='model_b',
                         reset_precision_policy=RESET, **kwargs)

    def select(self, horizon):
        key = base.scope_key('model_b', horizon)
        if key in self.state['selection']:
            return self.state['selection'][key]
        reference = base.read(self.root/'references'/f'{key}-calibration.json')['estimates']
        candidates = []
        for index, controls in enumerate(base.CONTROLS):
            result, out = self.numerical_task(f'{key}-cal-{index}', 'probe', 1200,
                horizon=horizon, controls=controls, partition='calibration')
            if result['status'] == 'PROBE_VALID':
                candidates.append(dict(index=index, controls=controls,
                    loss=nomination_loss(result['rows'][-1], reference), result=str(out/'result.json')))
        for candidate in sorted(candidates, key=lambda x: x['loss']):
            result, out = self.numerical_task(f'{key}-validation-{candidate["index"]}',
                'probe', 1200, horizon=horizon, controls=candidate['controls'], partition='validation')
            if result['status'] != 'PROBE_VALID':
                continue
            data, design = base.prepared_paths(self.root, 'model_b', horizon, 'final')
            selection = dict(model='model_b', horizon=horizon, controls=candidate['controls'],
                reset_precision_policy=RESET, particles=4032, dtype='float32', tf32=False,
                data_sha256=base.hash_file(data), design_sha256=base.hash_file(design),
                source_identity=base.source_identity(), calibration=candidates,
                validation=str(out/'result.json'), approximation_promoted=False,
                selection_rule='Descriptive calibration loss nominates; validation checks numerical health',
                statistically_supported_ranking=False)
            path = self.root/'selections'/f'{key}.json'
            base.save(path, selection)
            self.state['selection'][key] = str(path)
            self.persist()
            return str(path)
        return None

    def train(self, selection_path):
        selection = base.read(selection_path)
        common = dict(horizon=50, controls=selection['controls'], selection=selection_path)
        arms = []
        for width, rate in ((18, .001), (36, .0003)):
            for seed in (17, 29):
                key = f'B-train-w{width}-s{seed}'
                result, out = self.numerical_task(key, 'training', 5400, **common,
                    width=width, rate=rate, seed=seed, updates=256)
                if result.get('status') == 'TRAINING_INCOMPLETE' and result.get('checkpoint'):
                    result, out = self.numerical_task(key+'-resume', 'training', 2700, **common,
                        width=width, rate=rate, seed=seed, updates=256, resume=result['checkpoint'])
                if result.get('status') == 'TRAINING_CANDIDATE_VALID':
                    losses = result['heldout_losses']
                    arms.append(dict(width=width, rate=rate, seed=seed, result=result,
                                     mean_loss=sum(losses)/len(losses)))
        base.save(self.root/'training-nomination.json', dict(arms=arms,
            statistically_supported_ranking=False,
            rule='Common heldout ELBO nominates; downstream HMC determines utility'))
        if not arms:
            return None
        winner = min(arms, key=lambda x:x['mean_loss'])
        arguments = dict(**common, width=winner['width'], rate=winner['rate'], seed=winner['seed'], updates=1024)
        result, out = self.numerical_task('B-train-extend', 'training', 10800,
            **arguments, resume=winner['result']['checkpoint'])
        if result.get('status') == 'TRAINING_INCOMPLETE' and result.get('checkpoint'):
            result, out = self.numerical_task('B-train-extend-resume', 'training', 5400,
                **arguments, resume=result['checkpoint'])
        return result.get('frozen_map') if result.get('status') == 'TRAINING_CANDIDATE_VALID' else None

    def run(self, until='all'):
        self.recover()
        self.phase('prepare', 'Generate fresh T2/T50 calibration, validation and final data')
        result, _ = self.task('prepare', 'prepare', 300, gpu=False, scopes=SCOPES, seed_offset=SEED_OFFSET)
        if result['status'] != 'PREPARED':
            return self.stop('PREPARATION_REPAIR_REQUIRED', 'Inspect preparation error and retry within budget')
        self.phase('references', 'Independent short quadrature and long replicated particle references')
        result, _ = self.task('references', 'references', 6900, gpu=False, scopes=SCOPES)
        if result['status'] != 'REFERENCES_COMPUTED':
            return self.stop('REFERENCE_REPAIR_REQUIRED', 'Inspect independent reference failure before inference')
        if until == 'references':
            return self.stop('REFERENCES_COMPLETE_CAMPAIGN_PENDING', 'Resume GPU feasibility and tuning')
        self.phase('integration', 'Check native batch, analytical force, frozen map and shared XLA HMC')
        for kind, expected, seconds in [('probe', 'PROBE_VALID', 1200),
                ('training_setup', 'TRAINING_SETUP_VALID', 900), ('integration', 'INTEGRATION_VALID', 1200)]:
            result, _ = self.numerical_task('B-'+kind, kind, seconds, horizon=2, controls=base.CONTROLS[0])
            if result['status'] != expected:
                return self.stop('INTEGRATION_REPAIR_REQUIRED', f'Inspect {kind} and repair its failed invariant')
        if until == 'pilot':
            return self.stop('PILOT_COMPLETE_CAMPAIGN_PENDING', 'Resume scope tuning and fresh NeuTra training')
        self.phase('scope_tuning', 'Calibrate controls separately at T2 and T50')
        for _, horizon in SCOPES:
            selection_path = self.select(horizon)
            if selection_path is None:
                return self.stop('SCOPE_TUNING_REPAIR_REQUIRED', 'Inspect invalid candidates; no posterior promoted')
            selection = base.read(selection_path)
            result, out = self.numerical_task(f'B-T{horizon}-final', 'probe', 1200,
                horizon=horizon, controls=selection['controls'], partition='final')
            if result['status'] != 'PROBE_VALID':
                return self.stop('FINAL_TARGET_REPAIR_REQUIRED', 'Repair final target numerical validity before training')
            self.state.setdefault('final_probes', {})[str(horizon)] = str(out/'result.json')
            self.persist()
        self.report()
        self.phase('neutra_training', 'Fresh Model B capacity/rate/seed protocol and frozen-map checks')
        selection_path = self.state['selection']['model_b-T50']
        frozen = self.state.get('frozen_map') or self.train(selection_path)
        if not frozen:
            return self.stop('TRAINING_INCOMPLETE', 'Inspect training checkpoints and remaining budget; do not promote')
        self.state['frozen_map'] = frozen
        self.persist()
        self.phase('adaptive_hmc', 'Adapt warmup then retain inference draws to cumulative Rhat <=1.01 or caps')
        selection = base.read(selection_path)
        arguments = dict(horizon=50, controls=selection['controls'], selection=selection_path, frozen_map=frozen)
        result, out = self.numerical_task('B-hmc', 'hmc', 24000, **arguments)
        if not result.get('posterior_sampling_complete') and result.get('manifest') and self.remaining(True) >= 60:
            result, out = self.numerical_task('B-hmc-resume', 'hmc', 18000,
                **arguments, resume_manifest=result['manifest'])
        self.state['hmc'] = dict(result=result, output=str(out) if out else None)
        self.persist()
        self.phase('analysis', 'Report inference separately from independent approximation accuracy')
        self.report()
        done = result.get('posterior_sampling_complete', False)
        return self.stop('SAMPLING_COMPLETE_ACCURACY_UNRESOLVED' if done else 'SAMPLING_INCOMPLETE',
            'Review reference uncertainty and posterior table' if done else 'Inspect sampler checkpoint and budget before continuation')

    def report(self):
        comparisons = {}
        for horizon, path in self.state.get('final_probes', {}).items():
            reference = base.read(self.root/'references'/f'model_b-T{horizon}-final.json')
            comparisons[horizon] = compare_probe(base.read(path)['rows'][-1], reference)
        hmc = self.state.get('hmc', {}).get('result', {})
        report = dict(comparisons=comparisons, hmc=hmc,
            engineering='Native structural completion and shared HMC checked separately',
            statistical_ranking='Not established; no replicated LEDH superiority claim',
            approximation_accuracy='Unresolved: replicated PF uncertainty does not remove finite-particle bias',
            default_readiness=False, exact_posterior_claim=False,
            remaining_gpu_hours=self.remaining(True)/3600)
        base.save(self.root/'analysis.json', report)
        lines = ['# Structural tanh Model B confirmation', '',
            'Fresh data; FP32, 4032 particles, conditional FP64 reset. The finite LEDH likelihood is the HMC endpoint.', '',
            '| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |',
            '|---|---|---|---|---|---|',
            f'| Sampling | {hmc.get("posterior_sampling_complete", "pending")} | {hmc.get("status", "pending")} | Finite likelihood approximation | Read reference and posterior diagnostics | Exact posterior accuracy |', '',
            '| Inference status | Finding |', '|---|---|',
            '| Hard veto screen | Per-worker numerical validity; failures remain explicit in state.json |',
            '| Statistically supported ranking | None |',
            '| Descriptive differences | analysis.json contains likelihood, score, MCSE and refinement at every probe |',
            '| Default readiness | No default change |',
            '| Next evidence | Resolve material reference discrepancies before claiming exact-model accuracy |', '',
            'Post-run alternative explanation: mixing can be good for a biased finite likelihood. Weakest evidence is long-series independent score bias. A refined reference changing beyond its measured uncertainty would overturn the accuracy interpretation.']
        if hmc.get('physical_mean'):
            lines += ['', '| Parameter | Truth | Posterior mean | 95% interval |', '|---|---:|---:|---|']
            for i, name in enumerate(('rho', 'sigma', 'beta')):
                lines.append(f'| {name} | {hmc["truth"][i]:.6g} | {hmc["physical_mean"][i]:.6g} | [{hmc["credible_interval"][0][i]:.6g}, {hmc["credible_interval"][1][i]:.6g}] |')
        (self.root/'result.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'resume', 'status'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-hours', type=float)
    parser.add_argument('--until', choices=('references', 'pilot', 'all'), default='all')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.command == 'status':
        state = base.read(root/'state.json')
        print(json.dumps({key:state[key] for key in ('phase', 'decision', 'next_action', 'updated')}, indent=2))
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Confirmation(root, args.gpu_hours if args.command == 'run' else None)
        campaign.run(args.until)


if __name__ == '__main__':
    main()
