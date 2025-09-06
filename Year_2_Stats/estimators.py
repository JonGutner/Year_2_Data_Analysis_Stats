#Contains the different methods that perform estimation operations
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
from . import helpers
from . import pdfs

def guess_initial_params(data, pdf):
    """Automatic initial guesses for different PDFs."""
    mean = np.mean(data)
    std = np.std(data)

    if pdf == pdfs.gaussian:
        return [mean, std]
    elif pdf == pdfs.exponential:
        return [1/mean] if mean > 0 else [1.0]
    elif pdf == pdfs.poisson_pmf:
        return [mean]
    elif pdf == pdfs.binomial:
        n_guess = np.max(data)
        p_guess = mean / n_guess if n_guess > 0 else 0.5
        return [p_guess]
    elif pdf == pdfs.lorentzian:
        q25, q75 = np.percentile(data, [25, 75])
        return [np.median(data), q75 - q25]
    elif pdf == pdfs.uniform_pdf:
        return [np.min(data), np.max(data)]
    else:
        raise ValueError("No guess implemented for this PDF")

def mle_fit(data, pdf, init_params=None, method="Nelder-Mead"):
    """
    General-purpose MLE with auto parameter guessing and uncertainties.
    """
    if init_params is None:
        init_params = guess_initial_params(data, pdf)

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
