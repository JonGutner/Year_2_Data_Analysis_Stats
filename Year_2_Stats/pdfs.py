#Contains a list of different possible Probability Distribution Functions.
import numpy as np
from scipy.stats import norm, expon, poisson, binom, uniform

# =======Continuous PDFs=======
# Gaussian (Normal)
def gaussian(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

# Exponential (decay, lifetime distributions)
def exponential(x, lamb):
    return expon.pdf(x, scale=1/lamb)

# Uniform (flat background, quantization error)
def uniform_pdf(x, a, b):
    return uniform.pdf(x, loc=a, scale=(b - a))

# Lorentzian (Cauchy, resonance lines)
def lorentzian(x, x0, gamma):
    return (1/np.pi) * (0.5*gamma) / ((x - x0)**2 + (0.5*gamma)**2)

# =======Discrete PMFs=======
# Poisson (counts, radioactive events)
def poisson_pmf(k, mu):
    return poisson.pmf(k, mu)

# Binomial (success/failure counts)
def binomial_fixed_n(n):
    def pdf(k, p):
        return binom.pmf(k, n, p)
    return pdf

# ==Thermal Waves Experiment==
def sine_with_phase(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/10 * t - phase) + c

def phase_waves(x, m ,c):
    x = np.array(x, dtype=float)
    return m*x + c

def amplitude_waves(x, a, m ,c):
    x = np.array(x, dtype=float)
    return a*np.exp(m*x) + c