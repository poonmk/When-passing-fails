"""
Basal model
Consideration of only frequency (rate) of correct responses
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
print(f"Running on PyMC3 v{pm.__version__}")

# Test data set and conversion to rates (yp = frequency of correct responses)
y = np.asarray([27,26,23,34,27,31,25,27,28,33,30,30,31,35,21,27,29,31,26,25,27,30,22,31,27,32,27,28,35,34,31,25,28,30,33,32,30,28,26,24,31,25])
print('Length of array: ', y.size)
q = 40 # How many total questions
yp = y.astype(float)/q
print('marks: ', y)
print('Rates (yp): ', yp)

# Fit distribution to beta distribution maximum likelihood
MLE = stats.beta.fit(yp, floc=0, fscale=1) # Fix loc and scale
print('MLE pararms (alpha, beta, ...): ', MLE)
alpha,beta = MLE[0], MLE[1]
x = np.linspace(0,1,101)
plt.xlabel('p')
plt.ylabel('Count')
plt.title('MLE fit')
plt.hist(yp, density=True, label = 'x/q')
plt.plot(x,stats.beta.pdf(x, alpha, beta), label = 'beta fit')
plt.legend(loc='best')

# MCMC
niter = 50000 # Number of iterations
model = pm.Model() 

with model: # context management

    # define priors
    p = pm.Beta('p', alpha=alpha, beta=beta) # Use MLE fit as prior

    # Likelihood (sampling distribution) of observations
    obs = pm.Binomial('obs', n=q, p=p, observed=y)

    # inference
    trace = pm.sample(niter, return_inferencedata=False, target_accept=0.9)

    # Display results
    az.plot_trace(trace)
    az.plot_posterior(trace, hdi_prob=0.95)
    print('Fitted p = ',trace['p'].mean())
    az.summary(trace)
