#Contains the different methods that perform estimation operations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import numdifftools as nd
from Year_2_Stats import helpers, pdfs

from scipy.linalg import pinvh


def mle_fit(data, nll_func, init_params=None, method="BFGS", is_pdf=True):
    if init_params is None:
        raise ValueError("Please provide init_params for the fit.")

    # --- Minimize negative log-likelihood ---
    result = minimize(nll_func, init_params, method=method)
    best_params = result.x

    # --- Compute Hessian at best fit ---
    try:
        hessian = nd.Hessian(lambda p: nll_func(p))(best_params)
        hessian = 0.5 * (hessian + hessian.T)  # symmetrize
        try:
            cov = np.linalg.inv(hessian)
            fisher_errors = np.sqrt(np.abs(np.diag(cov)))
            method_used = "inv"
        except np.linalg.LinAlgError:
            cov = pinvh(hessian)
            fisher_errors = np.sqrt(np.abs(np.diag(cov)))
            method_used = "pinvh"
    except Exception as e:
        print("Warning: Hessian calculation failed:", e)
        fisher_errors = [np.nan] * len(best_params)
        method_used = "none"

    # --- Profile likelihood intervals (optional, requires helpers.profile_scan) ---
    try:
        from Year_2_Stats import helpers
        profile_errors = []
        for i in range(len(best_params)):
            low, high = helpers.profile_scan(i, best_params, data, nll_func if is_pdf else lambda d, *p: nll_func(p))
            profile_errors.append((low, high))
    except Exception:
        profile_errors = [(np.nan, np.nan)] * len(best_params)

    return {
        "params": best_params,
        "neg_logL": result.fun,
        "success": result.success,
        "method_used": method_used,
        "fisher_errors": fisher_errors,
        "profile_intervals": profile_errors
    }

def auto_goodness_of_fit(x, y, model, params, is_pdf=False):
    if is_pdf:
        return goodness_of_fit(y, model, params)
    else:
        return goodness_of_fit_regression(x, y, model, params)

def goodness_of_fit_regression(x, y, model, params, sigma=None):
    y_pred = model(x, *params)
    residuals = y - y_pred

    if sigma is None:
        sigma = np.std(residuals)

    chi2_val = np.sum((residuals / sigma) ** 2)
    dof = len(y) - len(params)
    p_value = 1 - chi2.cdf(chi2_val, dof)

    return {"chi2": chi2_val, "dof": dof, "p_value": p_value, "sigma": sigma}

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
