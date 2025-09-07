#Contains the different methods that perform estimation operations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import numdifftools as nd
from Year_2_Stats import helpers, pdfs

def mle_fit(data, pdf, init_params=None, method="Nelder-Mead"):
    """
    General-purpose MLE with auto parameter guessing and uncertainties.
    """
    if init_params is None:
        init_params = helpers.guess_initial_params(data, pdf)

    result = minimize(helpers.neg_log_likelihood, init_params,
                      args=(data, pdf),
                      method=method)

    best_params = result.x

    # Errors via Hessian
    hessian = nd.Hessian(lambda p: helpers.neg_log_likelihood(p, data, pdf))(best_params)
    cov = np.linalg.inv(hessian)
    fisher_errors = np.sqrt(np.diag(cov))

    # Profile scans for each parameter
    profile_errors = []
    for i in range(len(best_params)):
        low, high = helpers.profile_scan(i, best_params, data, pdf)
        profile_errors.append((low, high))

    return {
        "params": best_params,
        "neg_logL": result.fun,
        "success": result.success,
        "fisher_errors": fisher_errors,
        "profile_intervals": profile_errors
    }

def goodness_of_fit(data, pdf, params, bins=50):
    """
    Chi-squared goodness-of-fit test.
    Handles continuous (Gaussian, Exponential, Lorentzian, Uniform, …)
    and discrete (Poisson, Binomial) distributions.
    """
    is_poisson = pdf.__name__ == "poisson_pmf"
    is_binomial = pdf.__name__ == "pdf" and pdf.__closure__ is not None
    is_discrete = is_poisson or is_binomial

    if is_discrete:
        kmin = int(np.min(data))
        kmax = int(np.max(data))
        k_vals = np.arange(kmin, kmax + 1)

        obs_counts, _ = np.histogram(data, bins=np.arange(kmin - 0.5, kmax + 1.5, 1))
        exp_probs = pdf(k_vals, *params)
        exp_counts = exp_probs * len(data)

        mask = exp_counts >= 5
        obs_counts = obs_counts[mask]
        exp_counts = exp_counts[mask]

    else:
        obs_counts, edges = np.histogram(data, bins=bins)
        exp_probs = []
        for i in range(len(obs_counts)):
            left, right = edges[i], edges[i+1]
            mid = 0.5 * (left + right)
            prob = pdf(mid, *params) * (right - left)
            exp_probs.append(prob)
        exp_probs = np.array(exp_probs)
        exp_counts = exp_probs * len(data)

        mask = exp_counts >= 5
        obs_counts = obs_counts[mask]
        exp_counts = exp_counts[mask]

    chi2_val = np.sum((obs_counts - exp_counts) ** 2 / exp_counts)
    dof = len(obs_counts) - len(params) - 1
    if dof <= 0:
        dof = 1
    p_val = 1 - chi2.cdf(chi2_val, dof)

    return {"chi2": chi2_val, "dof": dof, "p_value": p_val}
