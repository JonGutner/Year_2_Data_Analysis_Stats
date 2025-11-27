#Contains helper methods
import numpy as np
import glob, os
import pandas as pd
from Year_2_Stats import estimators, outputer, pdfs

def load_data(folder):
    # current_dir points to StatsToolBox/Year_2_Stats
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # go up one level to StatsToolBox/, then into Data_1
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    folder_path = os.path.join(base_dir, folder)

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    data_frames = [pd.read_csv(file) for file in csv_files]
    return data_frames

def neg_log_likelihood(params, data, pdf):
    """Generic negative log likelihood for MLE."""
    vals = pdf(data, *params)
    vals = np.clip(vals, 1e-12, None)  # avoid log(0)
    return -np.sum(np.log(vals))

def profile_scan(param_idx, params, data, pdf, step=0.05, n_steps=40):
    """
    Profile likelihood scan for 1 parameter.
    Returns (low, high) bounds for ~68% CL.
    """
    ll0 = -neg_log_likelihood(params, data, pdf)
    scan_vals = []

    for shift in np.linspace(-n_steps*step, n_steps*step, 2*n_steps+1):
        trial = params.copy()
        trial[param_idx] += shift
        ll = -neg_log_likelihood(trial, data, pdf)
        scan_vals.append((trial[param_idx], ll))

    # Extract interval (ΔlnL = 0.5)
    ci = [val for val, ll in scan_vals if ll >= ll0 - 0.5]
    if ci:
        return min(ci), max(ci)
    return np.nan, np.nan

def run_tests_pdf(data_frames, chosen_pdf, param_names, folder):
    # Loop over datasets
    for i in range(len(data_frames)):
        df = data_frames[i]
        # If 1 column, use directly; if 2+, take second column as y-values
        if df.shape[1] == 1:
            data = df.iloc[:, 0].dropna().to_numpy()
        else:
            data = df.iloc[:, 1].dropna().to_numpy()

        # Perform MLE
        result = estimators.mle_fit_pdf(data, chosen_pdf)

        # Output results + plot
        outputer.print_results(f"Dataset {i}", result, param_names, data=data, pdf=chosen_pdf)
        outputer.show_fit(data, chosen_pdf, result["params"], folder,
                          title=f"{i}")

def guess_initial_params(data, pdf):
    mean = np.mean(data)
    std = np.std(data)

    if pdf == pdfs.gaussian:
        return [mean, std]
    elif pdf == pdfs.exponential:
        return [1/mean] if mean > 0 else [1.0]
    elif pdf == pdfs.poisson_pmf:
        return [mean]
    elif pdf == pdfs.lorentzian:
        q25, q75 = np.percentile(data, [25, 75])
        return [np.median(data), q75 - q25]
    elif pdf == pdfs.uniform_pdf:
        return [np.min(data), np.max(data)]
    else:
        raise ValueError("No guess implemented for this PDF")