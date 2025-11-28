import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2
import numdifftools as nd
from scipy.linalg import pinvh
from Waves_Lab import w_pdfs
from Year_2_Stats import helpers

def mle_fit(data, nll_func, init_params=None, method="TNC", is_pdf=True):
    if init_params is None:
        raise ValueError("Please provide init_params for the fit.")

    # --- PARAMETER BOUNDS ---
    # amplitude >= 0
    # phase in [-pi, pi]
    # offset unbounded
    bounds = [(0, None), (-2*np.pi, 2*np.pi), (None, None)]

    # --- Minimize negative log-likelihood ---
    result = minimize(
        nll_func,
        init_params,
        method=method,
        bounds=bounds,                # <<< ADDED
        options={"maxiter": 10000}    # <<< optional but helpful
    )

    best_params = result.x

    # --- Compute Hessian, Fisher errors (your code unchanged) ---
    try:
        hessian = nd.Hessian(lambda p: nll_func(p))(best_params)
        hessian = 0.5 * (hessian + hessian.T)
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

    # profile likelihood unchanged
    try:
        profile_errors = []
        for i in range(len(best_params)):
            low, high = helpers.profile_scan(i, best_params, data,
                   nll_func if is_pdf else lambda d, *p: nll_func(p))
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

def goodness_of_fit_regression(x, y, model, params, sigma=None):
    y_pred = model(x, *params)
    residuals = y - y_pred

    if sigma is None:
        sigma = np.std(residuals)

    chi2_val = np.sum((residuals / sigma) ** 2)
    dof = len(y) - len(params)
    p_value = 1 - chi2.cdf(chi2_val, dof)

    return {"chi2": chi2_val, "dof": dof, "p_value": p_value, "sigma": sigma}

def amplitude_fit(spread, amplitude, guess_vals):
    return curve_fit(w_pdfs.amplitude_waves, spread, amplitude, guess_vals)

def phase_fit(spread, phase, guess_vals):
    return curve_fit(w_pdfs.phase_waves, spread, phase, guess_vals)