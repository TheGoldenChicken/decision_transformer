import numpy as np

"""
Calculating normalized scores
"""

#Half, hopper, walker
SAC = np.array([12135.0, 3234.3, 4592.3])
random = np.array([-288.8, 18.4, 1.9])

# Mean values for all seeds at the best performing iterations
score = np.array([[4127.651609104094,
                    4229.820876225356,
                    4312.880383162523,
                    4246.662206565949,
                    4337.778615555158],
                    [2802.8026698439785,
                    2785.5594905079065,
                    2419.9395951328443,
                    2891.8472652769533,
                    2301.8498854645704],
                    [2968.9145344087365,
                    3479.3699707560863,
                    3251.908267501702,
                    3053.1594628374023,
                    2977.5072518240545]]).T

# Calculating and prining normalized scores
normalized = 100*(score-random)/(SAC-random)
standardized_mean = np.mean(normalized,axis=0)
standardized_std = np.std(normalized,axis=0)
mean = np.mean(score,axis=0)
std = np.std(score,axis=0)
print(standardized_mean,standardized_std) #normalized
print(mean,std) #non-normalized

"""
Doing statistical 1-sample ttest
"""
from scipy import stats

d4rl_mean = np.array([3093.3,467.3,682.7])

results = stats.ttest_1samp(score,d4rl_mean, axis=0)
print("pvalues:",results.pvalue)
print("meandiffs:",mean-d4rl_mean)








# standardized = 100*((score-random)/(SAC-random))