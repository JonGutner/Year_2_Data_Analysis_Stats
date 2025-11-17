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
        outputer.show_fit_pdf(data, chosen_pdf, result["params"], folder,
                          title=f"{i}")

# ----------------------------
# Updated run_tests_waves with DC offset
# ----------------------------
def run_tests_waves(df, i, chosen_pdf, param_names, j):
    """
    Perform MLE fit for a single dataset (df) and print/plot results.
    Handles standard PDFs and sine_with_phase with DC offset.
    """
    # Extract time (t) and measured values (y)
    t = df.iloc[:, 0].dropna().to_numpy()
    y = df.iloc[:, 1].dropna().to_numpy()

    # --- Sine-wave with DC offset ---
    def sine_nll(params):
        amplitude, phase, offset = params
        model = chosen_pdf(t, amplitude, phase, offset)
        residuals = y - model
        sigma = 1.0  # assumed measurement error
        return 0.5 * np.sum((residuals / sigma) ** 2)

    # Initial guess: amplitude ~ max-min, phase ~0, offset ~ mean(y)
    init_params = [0.5*(np.max(y)-np.min(y)), np.pi/2, np.mean(y)]

    # Perform MLE
    result = estimators.mle_fit_waves(y, sine_nll, init_params=init_params, method="BFGS", is_pdf=False)

    # Plot function including DC offset
    plot_pdf = lambda x, amplitude, phase, offset: chosen_pdf(x, amplitude, phase, offset)

    # Print results
    outputer.print_results_waves(t, y, result, i, j, param_names, pdf=chosen_pdf)

    # Show fit using original time array
    outputer.show_fit_waves(y, plot_pdf, result["params"], j, t=t, title=f"Therm_{i}")

    return result["params"], result["fisher_errors"]

def run_waves_plots(packages):
    y_models_a = []
    y_models_p = []
    d = 0.05
    spacing = [0*d, 1*d, 2*d, 3*d, 4*d, 5*d]

    for i in range(4):
        package = packages[f"package_{i}"]
        amplitude = package[0]
        phase = package[1]

        # print ("======>", phase)

        params_a = [amplitude[0], -1, 0]
        popt_a, pcov_a = estimators.amplitude_fit_waves(spacing, amplitude, params_a)
        y_models_a.append(pdfs.amplitude_waves(spacing, *popt_a))

        params_p = [0.3, 0]
        popt_p, pcov_p = estimators.phase_fit_waves(spacing, phase, params_p)
        y_models_p.append(pdfs.phase_waves(spacing, *popt_p))

    outputer.show_thermistor_param(spacing, packages, y_models_a, y_models_p)

def get_ampli_phase_err_waves(df_0, df_1, df_2, df_3, df_4, df_5, chosen_pdf, param_names, j):
    amplitudes = []
    phases = []
    err_a = []
    err_p = []

    results, errs = run_tests_waves(df_0, 0, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests_waves(df_1, 1, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests_waves(df_2, 2, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests_waves(df_3, 3, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    #EXTRA THERMISTORS:
    results, errs = run_tests_waves(df_4, 4, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    results, errs = run_tests_waves(df_5, 5, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    return amplitudes, phases, err_a, err_p

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