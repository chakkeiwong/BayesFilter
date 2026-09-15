# Source-support ledger

| Source | Technical material checked | Supported use here | Boundary |
|---|---|---|---|
| Zhao & Cui, *Tensor-train methods for sequential state and parameter learning in state-space models*, JMLR 25 (2024), local PDF/text and official page | Algorithm 2 target (Eq. 15), conditional update and correction (Eqs. 21--23), Section 5.1 bridge/map/ratio (Eqs. 30--33), Section 5.2 Gaussian bridge, Section 5.3 tempering | Exact `f*g` correction; bridge-coordinate construction; Gaussian and tempered baselines | Does not establish UKF superiority or a local implementation's call-chain correctness |
| Zhao--Cui author MATLAB source, `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m` and `full_sol.m` | Propagate then likelihood-weight; weighted Gaussian factor; transformed log target with transition/likelihood terms and determinant | Confirms practical author route and exact-factor retention | Author code is an audit anchor, not an oracle for the TensorFlow lane |
| Julier & Uhlmann, *A New Extension of the Kalman Filter to Nonlinear Systems* (1997), local PDF | `2n+1` sigma-point unscented transform; predicted observation covariance and state-observation cross-covariance | Definition of standard UKF comparator | No claim that UKF is optimal for this model |
| Cui & Dolgov, *Deep composition of tensor-trains using squared inverse Rosenblatt transports* (2022), local PDF | Bridge densities, composed transport, ratio-based preconditioning and error implications | Supports transport/bridge interpretation and the need to fit the transformed ratio | Does not validate the local row implementation or establish filtering performance |
| Cohen & Migliorati, *Optimal weighted least-squares methods* (2017), local PDF | Sampling measure/weight relation and Christoffel stability | Supports recomputing row laws and weights after a coordinate/reference change | Stability theorem does not certify the current finite TT rank or rows |

## Discovery and omission record

The exact Zhao--Cui paper was checked locally and against the official JMLR metadata page. Forward citation lookup through OpenAlex was attempted; it did not yield a usable technical successor set in the bounded pass. No claim depends on citation counts. The next literature pass should search successors specifically for transport filtering, sigma-point proposal adaptation, and conditional TT/KR maps before any default change.
