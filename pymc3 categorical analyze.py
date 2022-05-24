"""
Model with with paired confidence
Input confidence (feel_raw) and test performance (result) as array
Array is arranged in order oconfidence: [most confident (know), intermediate confidence (hunch), least confident (guess)]
"""

import numpy as np
import pymc3 as pm
import arviz as az
print(f"Running on PyMC3 v{pm.__version__}")

# Test data set and conversion to rates (yp)
feel_raw = ([34,  5,  1],[24,  7,  9],[12, 21,  7],[19, 14,  7],[26, 10,  4],[37,  3,  0],[22, 13,  5],[28, 10,  2],[27, 12,  1],[27, 10,  3],[35,  5,  0],[ 8, 21, 11],[20, 15,  5],[24, 16,  0],[25, 15,  0],[10, 24,  6],[23, 13,  4],[20, 11,  9],[33,  5,  2],[28, 10,  2],[20, 14,  6],[28,  9,  3],[12, 11, 17],[24, 14,  2],[20, 11,  9],[ 8, 25,  7],[26, 10,  4],[ 9, 25,  6],[26,  8,  6],[21, 16,  3],[31,  9,  0],[11, 19, 10],[23, 12,  5],[13, 15, 12],[10,  7, 23],[27,  8,  5],[13, 23,  4],[33,  4,  3],[12, 23,  5],[26, 12,  2],[28, 10,  2],[30, 10,  0],[24, 13,  3],[12, 16, 12],[23, 15,  2],[25,  8,  7],[24, 13,  3],[24, 14,  2],[14, 14, 12],[14, 18,  8],[20, 19,  1],[ 9, 22,  9],[19, 15,  6],[14, 24,  2],[30,  8,  2],[24, 12,  4],[16, 11, 13],[15, 13, 12])
feel_raw = np.asarray(feel_raw) # instances per category of confidence (most to least)
result = ([28,  2,  0],[18,  4,  4],[10,  8,  1],[15,  6,  1],[22,  8,  0],[36,  3,  0],[21, 10,  4],[22,  7,  0],[22,  8,  0],[27,  8,  2],[35,  4,  0],[ 7,  9,  4],[18,  5,  2],[21,  8,  0],[24,  5,  0],[10, 14,  3],[17,  9,  2],[16,  0,  4],[33,  3,  2],[23,  4,  0],[17,  8,  1],[25,  6,  2],[11,  8,  8],[16,  6,  0],[ 3,  4,  2],[ 7, 14,  0],[22,  7,  0],[ 5, 11,  2],[26,  7,  1],[16,  9,  0],[29,  4,  0],[11,  5,  3],[20, 10,  4],[11,  9,  3],[ 7,  4, 14],[14,  3,  2],[ 9, 19,  2],[30,  2,  0],[11, 13,  3],[25,  8,  0],[25,  7,  0],[22,  3,  0],[23,  4,  0],[10, 13,  7],[15,  8,  1],[20,  4,  3],[16,  8,  0],[23, 10,  2],[ 8,  7,  5],[11,  8,  3],[13,  8,  0],[ 7,  6,  1],[17, 10,  1],[11, 10,  1],[27,  5,  0],[23, 10,  2],[16,  9,  7],[11,  9,  6])
result = np.asarray(result) # test scores per confidence category
feel = np.sum(feel_raw, axis=0) # tally know, hunch, guess
cat = len(feel) # this many categories
N = int(feel.sum()) # Total # of questions over all students, as an array
Q = len(feel_raw) # Number of questions per test
print ("[confidence ratings tally], [test results tally by confidence]:", feel, np.sum(result, axis=0))
print ("# categories, # students, # questions:", cat, N, Q)

# MCMC
alpha, beta = 1,1
expected_guess = 0.2
niter = 50000

with pm.Model():

   # define priors
    b = pm.Dirichlet('b', a=np.ones(cat)) # Distributions of feelings: guess, hunch, know (prior)
    q = pm.Multinomial('q', n=N, p=b, observed=feel) # Number of questions guessed, hunched, or known; sum to N

    # Distributions of success probability for guess, hunch, and know
    p = pm.Beta('p', alpha=alpha, beta=alpha, shape=cat)

    # Likelihood (sampling distribution) of observations
    obs = pm.Binomial('obs', n=feel_raw, p=p, observed=result, shape=cat) # Each probability of success a binomial

    # Define calculated (deterministic) variables
    blunder_down = pm.Deterministic('blunder', 1 - p[0]) # compute unlucky blunder
    blunder_up = pm.Deterministic('blunder_up', p[2] - expected_guess) # optional: compute lucky blunder
    
    # inference   
    trace = pm.sample(draws=niter, return_inferencedata=False)
    
    # Display results
    az.plot_trace(trace)
    az.plot_posterior(trace, hdi_prob=0.95)
    az.summary(trace)
