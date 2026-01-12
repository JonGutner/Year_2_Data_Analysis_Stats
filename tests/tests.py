import unittest
import numpy as np
from Year_2_Stats import estimators, pdfs, outputer

class TestMLE(unittest.TestCase):
    def test_gaussian(self):
        true_mu, true_sigma = 5.0, 2.0
        n = 1000
        data = np.random.normal(true_mu, true_sigma, n)

        result = estimators.mle_fit_pdf(data, pdfs.gaussian)
        mu_est, sigma_est = result["params"]

        outputer.print_results(f"Dataset - Gaussian", result, ["mu", "sigma"],
                               data=data, pdf=pdfs.gaussian)
        outputer.show_fit(data, pdfs.gaussian, result["params"],
                          folder_suffix="Test_Runs", title="Gaussian")

        self.assertAlmostEqual(mu_est, true_mu, delta=0.10)   # tighter because n large
        self.assertAlmostEqual(sigma_est, true_sigma, delta=0.10)

    def test_exponential(self):
        true_lambda = 0.5
        n = 1000
        data = np.random.exponential(1/true_lambda, n)

        result = estimators.mle_fit_pdf(data, pdfs.exponential)
        lambda_est = result["params"][0]

        outputer.print_results(f"Dataset - Exponential", result, ["lambda"],
                               data=data, pdf=pdfs.exponential)
        outputer.show_fit(data, pdfs.exponential, result["params"],
                          folder_suffix="Test_Runs", title="Exponential")

        self.assertAlmostEqual(lambda_est, true_lambda, delta=0.03)

    def test_poisson(self):
        true_mu = 4.0
        n = 1000
        data = np.random.poisson(true_mu, n)

        result = estimators.mle_fit_pdf(data, pdfs.poisson_pmf)
        mu_est = result["params"][0]

        outputer.print_results(f"Dataset - Poisson", result, ["mu"],
                               data=data, pdf=pdfs.poisson_pmf)
        outputer.show_fit(data, pdfs.poisson_pmf, result["params"],
                          folder_suffix="Test_Runs", title="Poisson")

        self.assertAlmostEqual(mu_est, true_mu, delta=0.12)

    def test_binomial(self):
        n_trials = 20
        true_p = 0.3
        n = 1000
        data = np.random.binomial(n_trials, true_p, size=n)

        # Fix n, only fit p
        pdf = pdfs.binomial_fixed_n(n_trials)

        result = estimators.mle_fit_pdf(data, pdf, init_params=[0.5])
        p_est = result["params"][0]

        outputer.print_results(f"Dataset - Binomial", result, ["p"],
                               data=data, pdf=pdf)
        outputer.show_fit(data, pdf, result["params"],
                          folder_suffix="Test_Runs", title="Binomial")

        self.assertAlmostEqual(p_est, true_p, delta=0.03)

if __name__ == "__main__":
    unittest.main()
