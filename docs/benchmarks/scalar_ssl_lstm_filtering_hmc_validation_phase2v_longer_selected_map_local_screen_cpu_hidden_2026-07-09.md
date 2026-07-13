# Scalar SSL-LSTM Filtering HMC Validation Phase 2V - Longer Selected MAP-Local Screen

## Decision

- phase2v_longer_selected_map_local_screen_passed: `True`
- vetoes: `[]`
- selected_kernel: `{'num_leapfrog_steps': 2, 'step_size': 0.785, 'trajectory_length_L_times_epsilon': 1.57, 'phase2u_selected_candidate_index': 0}`
- acceptance_rate: `0.40625`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 2V result and draft/review scalar reference/posterior-agreement subplan

## Selected Kernel Row

- status: `passed_hard_vetoes`
- hard vetoes: `[]`
- initial: `{'u_new': [0.0, 0.0, 0.0, 0.0], 'value': -37.77528495512359, 'score': [-1.5860631377008547e-11, 3.829910920404347e-12, -1.7671080939180588e-11, -4.20571540645479e-12], 'score_norm': 2.4416858704074592e-11}`
- acceptance: `0.40625`
- samples summary: `{'shape': [128, 4], 'finite_sample_count': 128, 'nonfinite_sample_count': 0, 'first_u_new': [2.4530123996569206, -0.8736243482968524, -3.6612572361783946, 0.9743697213063183], 'final_u_new': [4.026178258336013, 2.8873573655361913, -3.5821729135904885, 1.9414262306644288], 'mean_u_new': [2.339329346843717, 0.6170706799855532, -2.158614641877545, 1.5823448512102032], 'std_u_new': [1.4446082251234498, 1.4433962914543734, 1.5630226197334784, 1.2645246074262342], 'max_abs_u_new': 5.960663881275316}`
- log accept: `{'values': [-4.586586154043487, 0.15864557070578456, -11.706794768470678, 0.3204012479246885, -0.10959885228236099, 0.04769887639590076, -2.6357185339310303, -50.21477533633085, 0.17095406826720794, 0.5268426116965282, -2.462951075799805, -0.592774023284459, 0.11857912735196485, -0.024887958630271045, -1.991149755286588, -0.06813704492555939, -0.17208524017861326, -3.0033113110594396, -27.427866499579615, -14.367695943200669, -30.788058538485583, 3.548802096494689, -49.63768311979974, -8.555199662053555, -3.542931280036354, 0.17607693448912554, -4.130852615318559, -0.5295471761156361, 0.029725356796376612, -20.485273839868384, -0.974519643374488, -0.05690226725032377, -0.3343128436822085, -0.18592291155367757, -1.7986544959100743, 0.3131509576834999, 0.2645132505705663, -35.11316992340947, -0.39395716037758577, 0.7347838321733645, -8.686497317627632, 0.4318907897307729, 0.5010902370341659, -0.59209446204687, -0.014086009834309965, -0.2873751948121357, -57.40630763818034, -0.06436166300152146, -2.9562286405702256, -52.06717625224308, -23.38259507345966, 0.07774952395099038, -30.579642104023343, -2.9059780574700342, -23.71678063686791, -15.017199017422627, -176.233941591232, -711.3278637023375, -7.548539501691926, -2844.7900680514995, -190.436546329042, -7.028550526409544, -4.869441880304155, 0.590536218455055, -0.03492751489588608, -0.2454823368778516, -0.12953824052869756, -17.335251859936953, -680.3688248104169, -12.935587507675125, -160.7383144596428, -2529.8174940290846, -5.3392711474661345, -14.806686754860593, -8.94182259369128, -17.6279151767015, 0.07696971105472383, -5.363787862710382, -4.631223876493189, -0.2309376503022733, -0.8973173517141934, 0.19992756505313602, -16.785764329148968, -2.764897983109745, 0.016040449122893374, -0.33837140211401984, -4.277365910922277, 0.055523996553092925, 0.40486207699688803, -0.26967069860233606, -0.18733375116475037, -104.6931909190424, 0.3167641632153244, -2.979709653065364, -0.12601613967491154, -0.26837841786263944, -20.085935471682518, -8.50590758008373, -107.5955731201216, -0.1647357431603761, -1.2190757207732528, -66.68267791035231, 0.9566715723588324, -0.07113745260419224, -0.895705666806522, -0.0001200321952400385, 0.1846122899189957, -2.8405052602270677, -12.781637669567706, -0.8642646424367952, -1.5332662608054588, -19.10084951002967, -16.008158329004672, 1.1408445055451764, -83.07394455529716, -1.0715351559666981, -0.7135202661722502, 0.9489218075852092, -4.603132693946106, -141.10615731675463, 1.8096516579321535, -5.081319433787236, -70.05371850107275, -66.2201866179015, -11.045448972249957, -3.905801213542942, -3290.2187092038007, -148.9544378530798], 'finite_count': 128, 'nonfinite_count': 0, 'max_abs_finite': 3290.2187092038007}`
- target log prob: `{'values': [-38.702367428941905, -38.66325265852469, -38.66325265852469, -38.166567216466994, -38.475515526947966, -38.355967270847145, -38.355967270847145, -38.355967270847145, -37.85051477370975, -37.89586180506182, -37.89586180506182, -38.191292694240744, -37.9144838799606, -38.432791393625074, -38.432791393625074, -38.24177265007311, -38.18191860876308, -40.048647082138935, -40.048647082138935, -40.048647082138935, -40.048647082138935, -38.62957475016494, -38.62957475016494, -38.62957475016494, -38.62957475016494, -38.38046242737107, -38.38046242737107, -38.38046242737107, -38.35402345488814, -38.35402345488814, -38.35402345488814, -38.92535139846817, -38.850837141233015, -39.5152689837883, -40.22337244970995, -39.261651905117205, -39.03839452227662, -39.03839452227662, -39.168229119012665, -39.05933204188673, -39.05933204188673, -38.75319871414189, -38.178732597133724, -38.178732597133724, -37.95402911177067, -39.05269456204034, -39.05269456204034, -38.25802207227614, -38.25802207227614, -38.25802207227614, -38.25802207227614, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.54373609580472, -38.13488113774926, -38.13488113774926, -38.13488113774926, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -38.45083340678095, -39.869561215923284, -39.869561215923284, -39.869561215923284, -38.99856087320798, -38.99856087320798, -39.5445644285839, -39.5445644285839, -39.5445644285839, -39.390085655691784, -39.03234754462776, -39.03234754462776, -38.609744909352855, -37.8872731848349, -37.8872731848349, -37.927018695111734, -37.927018695111734, -38.06165657747921, -38.06165657747921, -38.137135364924525, -38.137135364924525, -38.137135364924525, -38.137135364924525, -38.137135364924525, -38.527347030080556, -40.80353988356453, -40.80353988356453, -38.45798457218049, -38.34217247516388, -38.471951583469824, -38.579355440199514, -39.562113094077525, -39.562113094077525, -39.562113094077525, -39.562113094077525, -41.29766007404644, -41.29766007404644, -41.29766007404644, -40.48269400784244, -40.48269400784244, -40.48269400784244, -40.48269400784244, -39.9501095614332, -39.9501095614332, -39.9501095614332, -38.502603444204595, -38.502603444204595, -38.502603444204595, -38.502603444204595, -38.502603444204595, -38.502603444204595, -38.502603444204595, -38.502603444204595], 'finite': True, 'min': -41.29766007404644, 'max': -37.85051477370975}`
- native divergence: `{'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'}`

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| native_divergence | native divergence unavailable for at least one candidate; unavailable is not zero divergences |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none; single selected-kernel screen |
| descriptive_only_differences | acceptance, target-log-prob range, log-accept range, sample range, and runtime |
| posterior_correctness | not assessed |
| hmc_readiness | not assessed; Phase 2V finite/acceptance screen only |
| gpu_xla_readiness | blocked |
| default_readiness | not assessed |
| next_evidence_needed | reviewed scalar reference/posterior-agreement subplan |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md` |
| git | `{'commit': 'f297b103303c64019302ed5d9b9aaf2c8f919b64', 'dirty': True, 'dirty_line_count': 62, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M bayesfilter/linear/kalman_qr_tf.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/kalman_qr_analytic_vs_autodiff_score_scaling_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'cpu_hidden': True, 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| random_seeds | `[[20260709, 6401]]` |
| wall_time_seconds | `178.8504172930261` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md` |

## Nonclaims

- Phase 2V longer selected MAP-local finite/acceptance screen only
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
