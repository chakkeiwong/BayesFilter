# Matched LGSSM Recursive-Score Comparison

- hard_valid: `True`
- wall time: `209.3 s`
- peak GPU allocator: `474037248` bytes
- GenUT: verified bitwise-identical design alias of Cubature

## Kalman-Relative Error

| T | Method | Value | phi1 | phi2 | phi3 | q_scale | r_scale |
|---:|---|---|---|---|---|---|---|
| 2 | Contract E Gaussian residual | +0.077% [-1.461,+1.615]% | +2.051% [-9.483,+13.585]% | +3.064% [-24.601,+30.729]% | +15.542% [-70.121,+101.206]% | +0.815% [-12.204,+13.834]% | +1.601% [-6.867,+10.070]% |
| 2 | Cubature = Gaussian GenUT | +0.057% [-1.506,+1.620]% | +1.868% [-9.778,+13.514]% | -1.069% [-31.973,+29.835]% | +16.173% [-65.698,+98.045]% | +0.740% [-13.124,+14.603]% | +1.282% [-7.252,+9.816]% |
| 10 | Contract E Gaussian residual | -0.268% [-0.603,+0.067]% | +4.289% [-0.807,+9.385]% | -11.478% [-86.026,+63.071]% | -10.919% [-27.813,+5.974]% | +1.726% [-7.269,+10.721]% | +3.353% [-0.863,+7.570]% |
| 10 | Cubature = Gaussian GenUT | -0.386% [-0.750,-0.022]% | +5.311% [-0.447,+11.069]% | -20.774% [-82.970,+41.422]% | -7.407% [-24.469,+9.655]% | +6.534% [-1.631,+14.699]% | +4.595% [-1.412,+10.602]% |
| 50 | Contract E Gaussian residual | +0.116% [-0.135,+0.367]% | -10.378% [-22.236,+1.479]% | -1.767% [-16.399,+12.866]% | +35.547% [-25.025,+96.119]% | -28.561% [-124.966,+67.845]% | -3.717% [-57.624,+50.190]% |
| 50 | Cubature = Gaussian GenUT | -0.006% [-0.240,+0.227]% | -2.416% [-16.666,+11.834]% | -7.408% [-15.626,+0.809]% | +3.563% [-95.636,+102.763]% | +3.717% [-94.669,+102.102]% | +22.903% [-29.265,+75.071]% |

The JSON contains raw physical scores, physical score-error intervals,
and paired absolute-error delta intervals. Negative paired deltas favor
Cubature; positive paired deltas favor Gaussian-residual Contract E.
