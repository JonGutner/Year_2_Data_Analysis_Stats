#Contains helper methods
import numpy as np
import glob, os
import pandas as pd
from . import estimators, outputer

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
    from .helpers import neg_log_likelihood
    import numpy as np

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

def run_tests(data_frames, chosen_pdf, param_names):
    # Loop over datasets
    for i in range(len(data_frames)):
        df = data_frames[i]
        # If 1 column, use directly; if 2+, take second column as y-values
        if df.shape[1] == 1:
            data = df.iloc[:, 0].dropna().to_numpy()
        else:
            data = df.iloc[:, 1].dropna().to_numpy()

        # Perform MLE
        result = estimators.mle_fit(data, chosen_pdf)

        # Output results + plot
        outputer.print_results(f"Dataset {i}", result, param_names)
        outputer.show_fit(data, chosen_pdf, result["params"],
                          title=f"Dataset {i} fit")