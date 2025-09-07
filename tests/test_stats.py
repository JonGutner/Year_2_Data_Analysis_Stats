import unittest
import numpy as np
from Year_2_Stats import estimators, pdfs

class TestMLE(unittest.TestCase):
    def test_gaussian(self):
        np.random.seed(0)
        true_mu, true_sigma = 5.0, 2.0
        n = 1000
        data = np.random.normal(true_mu, true_sigma, n)

        result = estimators.mle_fit(data, pdfs.gaussian)
        mu_est, sigma_est = result["params"]

        self.assertAlmostEqual(mu_est, true_mu, delta=0.10)   # tighter because n large
        self.assertAlmostEqual(sigma_est, true_sigma, delta=0.10)

    def test_exponential(self):
        np.random.seed(1)
        true_lambda = 0.5
        n = 1000
        data = np.random.exponential(1/true_lambda, n)

        result = estimators.mle_fit(data, pdfs.exponential)
        lambda_est = result["params"][0]

        self.assertAlmostEqual(lambda_est, true_lambda, delta=0.03)

    def test_poisson(self):
        np.random.seed(2)
        true_mu = 4.0
        n = 1000
        data = np.random.poisson(true_mu, n)

        result = estimators.mle_fit(data, pdfs.poisson_pmf)
        mu_est = result["params"][0]

        self.assertAlmostEqual(mu_est, true_mu, delta=0.12)

    def test_binomial(self):
        np.random.seed(10)
        n_trials = 20
        true_p = 0.3
        n = 1000
        data = np.random.binomial(n_trials, true_p, size=n)

        # Fix n, only fit p
        pdf = pdfs.binomial_fixed_n(n_trials)

        result = estimators.mle_fit(data, pdf, init_params=[0.5])
        p_est = result["params"][0]

        self.assertAlmostEqual(p_est, true_p, delta=0.03)

    def test_lorentzian(self):
        np.random.seed(3)
        true_x0, true_gamma = 0.0, 1.0
        n = 1500
        # standard_cauchy has pdf with scale=1; scale it to gamma/2 to match our lorentzian paramization
        data = np.random.standard_cauchy(n) * (true_gamma/2) + true_x0

        result = estimators.mle_fit(data, pdfs.lorentzian)
        x0_est, gamma_est = result["params"]

        # Lorentzian MLEs can be noisier — allow a larger tolerance on gamma
        self.assertAlmostEqual(x0_est, true_x0, delta=0.12)
        self.assertAlmostEqual(gamma_est, true_gamma, delta=0.25)

if __name__ == "__main__":
    unittest.main()
