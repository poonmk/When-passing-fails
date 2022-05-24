"""
Model with k (explicit knowledge)
Consideration of bonly frequency (rate) of correct responses 
"""

import numpy as np
import pymc3 as pm
import arviz as az
print(f"Running on PyMC3 v{pm.__version__}")

# Test data set and conversion to rates (yp)
y = np.asarray([27,26,23,34,27,31,25,27,28,33,30,30,31,35,21,27,29,31,26,25,27,30,22,31
,27,32,27,28,35,34,31,25,28,30,33,32,30,28,26,24,31,25])
print('Length of array: ', y.size)
q = 40 # How many total questions
yp = y.astype(float)/q
print('marks: ', y)
print('Rates (yp): ', yp)

# Priors for p e.g., (alpha1,beta1) = (2.0,5.0) yields maximum at p = 0.2
alpha1 = 2.0
beta1 = 5.0
# Prior for k e.g., combination of alpha2=beta2 yields maximum at 50% knowledge, magnitudes determine dispersion
alpha2 = 3.3
beta2 = 3.3
# Number of iterations for MCMC
niter = 50000

with pm.Model(): # context management

    # define priors
    p = pm.Beta('p', alpha=alpha1, beta=beta1)
    k = pm.BetaBinomial('k', alpha=alpha2, beta=beta2, n=y)
    
    # Likelihood (sampling distribution) of observations
    obs = pm.Binomial('obs', n=q-k, p=p, observed=y-k)

    # inference
    trace = pm.sample(niter, return_inferencedata=False)

    az.plot_trace(trace)
    az.plot_posterior(trace, hdi_prob=0.95)
