#Contains a list of different possible Probability Distribution Functions.
import numpy as np
from scipy.stats import norm, expon, poisson, binom, uniform

# Gaussian (Normal)
def gaussian(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

# Exponential (decay, lifetime distributions)
def exponential(x, lamb):
    return expon.pdf(x, scale=1/lamb)

# Poisson (counts, radioactive events)
def poisson_pmf(k, mu):
    return poisson.pmf(k, mu)

# Binomial (success/failure counts)
def binomial_fixed_n(n):
    """Return a binomial PMF with fixed n, fitting only p."""
    def pdf(k, p):
        return binom.pmf(k, n, p)
    return pdf

# Uniform (flat background, quantization error)
def uniform_pdf(x, a, b):
    return uniform.pdf(x, loc=a, scale=(b - a))

# Lorentzian (Cauchy, resonance lines)
def lorentzian(x, x0, gamma):
    return (1/np.pi) * (0.5*gamma) / ((x - x0)**2 + (0.5*gamma)**2)
