"""TensorFlow-free static identity for the admitted KSC Gaussian-sum scope."""

KSC_GAUSSIAN_SUM_UKF_DATASET_ID = "zhao_cui_sv_ksc_gaussian_sum_ukf_T20"
KSC_GAUSSIAN_SUM_UKF_HORIZON = 20
KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP = 32
KSC_GAUSSIAN_SUM_UKF_SCOPE = (
    "KSC-UKF-gaussian-sum-mass-preserving-clustered-T20-cap32-v1"
)
KSC_GAUSSIAN_SUM_UKF_TARGET_SIGNATURE = (
    "727718ec8c4b4a68e2bc59c5f88d33be8e24cc4b77095f9197a360f6c9e7114d"
)
KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES = (
    "gamma_source_probit",
    "beta_source_probit",
)


__all__ = [
    "KSC_GAUSSIAN_SUM_UKF_COMPONENT_CAP",
    "KSC_GAUSSIAN_SUM_UKF_DATASET_ID",
    "KSC_GAUSSIAN_SUM_UKF_HORIZON",
    "KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES",
    "KSC_GAUSSIAN_SUM_UKF_SCOPE",
    "KSC_GAUSSIAN_SUM_UKF_TARGET_SIGNATURE",
]
