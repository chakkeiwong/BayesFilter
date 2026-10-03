# Original full-model calibration failures

The original flow grid failed validity checks. These are calibration observations, not untouched final comparisons. Every returned coordinate and matched Kalman score is preserved in calibration_scores.csv. Invalid zero returns are guard sentinels, not actual scores, and have no score-accuracy interpretation. calibration_likelihoods.csv preserves raw returned likelihoods. The intentional stop is documented in ../run-01-intentional-stop.json.
