"""Public dispatch for ordinary and typed TensorFlow HMC tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_tensorflow_tuning import (
    BOUND_RETAINED_HMC_ARCHIVE_SCHEMA,
    FOUR_CHAIN_ACCEPTANCE_SCHEMA,
    TENSORFLOW_HMC_TUNING_SCHEMA,
    BoundRetainedHMCArchiveConfig,
    BoundRetainedHMCArchiveResult,
    BoundRetainedHMCArchiveRunner,
    FourChainAcceptanceDecision,
    FourChainMeanBandAcceptancePolicy,
    TensorFlowHMCKernelTuningConfig,
    TensorFlowHMCKernelTuningResult,
    _run_tensorflow_hmc_tuning,
    build_retained_bound_hmc_archive_runner_from_tuning_result,
    load_tensorflow_hmc_tuning_result,
)
from bayesfilter.inference.tuning_contract import (
    HMCTuningRunnerBinding,
    require_active_hmc_tuning_route,
)


def tune_hmc_kernel(
    *,
    adapter: Any,
    initial_position: Any,
    config: Any = None,
    output_dir: str | Path | None = None,
    negative_hessian: Any | None = None,
    initial_covariance: Any | None = None,
    parameter_scales: Any | None = None,
    diagnostic_callback: Any | None = None,
    verification_checkpoint_writer_config: Any | None = None,
    runner_binding: HMCTuningRunnerBinding | None = None,
) -> Any:
    """Run the active public tuner selected by its typed configuration."""

    require_active_hmc_tuning_route("tune_hmc_kernel")
    if isinstance(config, TensorFlowHMCKernelTuningConfig):
        unsupported = {
            "negative_hessian": negative_hessian,
            "initial_covariance": initial_covariance,
            "diagnostic_callback": diagnostic_callback,
            "verification_checkpoint_writer_config": (
                verification_checkpoint_writer_config
            ),
        }
        supplied = tuple(name for name, value in unsupported.items() if value is not None)
        if supplied:
            raise ValueError(
                "TensorFlow tuning does not accept legacy host options: "
                + ", ".join(supplied)
            )
        if runner_binding is None:
            raise ValueError("TensorFlow tuning requires a repository-issued binding")
        return _run_tensorflow_hmc_tuning(
            adapter=adapter,
            initial_position=initial_position,
            config=config,
            output_dir=output_dir,
            parameter_scales=parameter_scales,
            runner_binding=runner_binding,
        )

    if runner_binding is not None:
        raise ValueError(
            "runner_binding is supported only with "
            "TensorFlowHMCKernelTuningConfig; ordinary HMCKernelTuningConfig "
            "uses the exact adapter score and BayesFilter's default TFP runner"
        )

    from bayesfilter.inference.hmc_kernel_tuning import _run_canonical_hmc_tuning

    return _run_canonical_hmc_tuning(
        adapter=adapter,
        initial_position=initial_position,
        config=config,
        output_dir=output_dir,
        negative_hessian=negative_hessian,
        initial_covariance=initial_covariance,
        parameter_scales=parameter_scales,
        diagnostic_callback=diagnostic_callback,
        verification_checkpoint_writer_config=verification_checkpoint_writer_config,
        runner_binding=runner_binding,
    )


__all__ = [
    "BOUND_RETAINED_HMC_ARCHIVE_SCHEMA",
    "BoundRetainedHMCArchiveConfig",
    "BoundRetainedHMCArchiveResult",
    "BoundRetainedHMCArchiveRunner",
    "FOUR_CHAIN_ACCEPTANCE_SCHEMA",
    "FourChainAcceptanceDecision",
    "FourChainMeanBandAcceptancePolicy",
    "TENSORFLOW_HMC_TUNING_SCHEMA",
    "TensorFlowHMCKernelTuningConfig",
    "TensorFlowHMCKernelTuningResult",
    "build_retained_bound_hmc_archive_runner_from_tuning_result",
    "load_tensorflow_hmc_tuning_result",
    "tune_hmc_kernel",
]
