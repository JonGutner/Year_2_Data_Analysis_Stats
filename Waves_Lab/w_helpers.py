import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from Waves_Lab import w_estimators, w_outputer, w_pdfs, data_management

def run_tests(df, i, chosen_pdf, param_names, j, init_params, close_fig=False, thermal=True):
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

    fig = None
    result = None

    # Initial guess: amplitude ~ max-min, phase ~0, offset ~ mean(y)
    if thermal:
        init_params = [0.5*(np.max(y)-np.min(y)), np.pi/2, np.mean(y)]

        # Perform MLE
        result = w_estimators.mle_fit(y, sine_nll, init_params=init_params, method="BFGS", is_pdf=False)

        # Plot function including DC offset
        plot_pdf = lambda x, amplitude, phase, offset: chosen_pdf(x, amplitude, phase, offset)

        # Print results
        w_outputer.print_results(t, y, result, i, j, True, param_names, pdf=chosen_pdf)

        # Show fit using original time array
        fig = w_outputer.show_fit(y, plot_pdf, result["params"], j, t=t, title=f"Therm_{i}")
    else:
        result = w_estimators.mle_fit(y, sine_nll, init_params=init_params, method="BFGS", is_pdf=False)

        # Print results
        w_outputer.print_results(t, y, result, i, j, False, param_names, pdf=chosen_pdf)
    if close_fig:
        plt.close(fig)

    return result["params"], result["fisher_errors"]

def run_thermal_plots(packages):
    y_models_a = []
    y_models_p = []
    popts_a = []
    pcovs_a = []
    popts_p = []
    pcovs_p = []
    d = 0.05
    spacing = [0*d, 1*d, 2*d, 3*d, 4*d, 5*d]

    for i in range(4):
        package = packages[f"package_{i}"]
        amplitude = package[0]
        phase = package[1]

        # print ("======>", phase)

        params_a = [amplitude[0], -1, 0]
        popt_a, pcov_a = w_estimators.amplitude_fit(spacing, amplitude, params_a)
        popts_a.append(popt_a)
        pcovs_a.append(pcov_a)
        y_models_a.append(w_pdfs.amplitude_waves(spacing, *popt_a))

        params_p = [0.3, 0]
        popt_p, pcov_p = w_estimators.phase_fit(spacing, phase, params_p)
        popts_p.append(popt_p)
        pcovs_p.append(pcov_p)
        y_models_p.append(w_pdfs.phase_waves(spacing, *popt_p))

    w_outputer.show_thermistor_param(spacing, packages, y_models_a, y_models_p)

    w_outputer.find_diffusivity(popts_a, pcovs_a, popts_p, pcovs_p, packages)

def get_ampli_phase_err_thermal(df_0, df_1, df_2, df_3, df_4, df_5, chosen_pdf, param_names, j):
    amplitudes = []
    phases = []
    err_a = []
    err_p = []

    results, errs = run_tests(df_0, 0, chosen_pdf, param_names, j, None, True)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests(df_1, 1, chosen_pdf, param_names, j, None,True)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests(df_2, 2, chosen_pdf, param_names, j, None, True)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests(df_3, 3, chosen_pdf, param_names, j, None, True)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    #EXTRA THERMISTORS:
    results, errs = run_tests(df_4, 4, chosen_pdf, param_names, j, None, True)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    results, errs = run_tests(df_5, 5, chosen_pdf, param_names, j, None, True)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    return amplitudes, phases, err_a, err_p

def get_ampli_phase_err_electrical(df, chosen_pdf, param_names, freq, is_in_phase=True):
    amplitudes = []
    phases = []
    err_a = []
    err_p = []
    results_list = []

    df_1 = pd.DataFrame({
        'time' : df['time'],
        'ch1' : df['ch1'],
    })
    df_2 = pd.DataFrame({
        'time' : df['time'],
        'ch2' : df['ch2'],
    })
    df_3 = pd.DataFrame({
        'time' : df['time'],
        'ch3' : df['ch3'],
    })
    if is_in_phase:
        init_params = [0.5*(np.max(df['ch1'])-np.min(df['ch1'])), 0, np.mean(df['ch1'])]
    else:
        init_params = [0.5*(np.max(df['ch1'])-np.min(df['ch1'])), np.pi, np.mean(df['ch1'])]

    results, errs = run_tests(df_1, freq, chosen_pdf, param_names, 0, init_params, False, False)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results_list.append(results)
    results, errs = run_tests(df_2, freq, chosen_pdf, param_names, 1, init_params, False, False)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results_list.append(results)
    results, errs = run_tests(df_3, freq, chosen_pdf, param_names, 2, init_params, False, False)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results_list.append(results)

    return amplitudes, phases, err_a, err_p, results_list

def load_run_thermal_vs_electrical(param_names, is_thermal=False):
    if is_thermal:
        # Load data for THERMAL Waves Experiment
        data_name_1 = "period_1.csv"
        df_0_1, df_1_1, df_2_1, df_3_1, df_4_1, df_5_1 = data_management.send_data_thermal(data_name_1)
        data_name_2 = "period_2.csv"
        df_0_2, df_1_2, df_2_2, df_3_2, df_4_2, df_5_2 = data_management.send_data_thermal(data_name_2)
        data_name_3 = "period_3.csv"
        df_0_3, df_1_3, df_2_3, df_3_3, df_4_3, df_5_3 = data_management.send_data_thermal(data_name_3)
        data_name_4 = "period_4.csv"
        df_0_4, df_1_4, df_2_4, df_3_4, df_4_4, df_5_4 = data_management.send_data_thermal(data_name_4)

        # -----------------------------
        # Run tests for THERMAL Waves Experiment
        amplitudes_1, phases_1, err_a_1, err_p_1 = (get_ampli_phase_err_thermal
                                                    (df_0_1, df_1_1, df_2_1, df_3_1, df_4_1, df_5_1,
                                                     w_pdfs.sine_with_phase_15, param_names, 0))
        amplitudes_2, phases_2, err_a_2, err_p_2 = (get_ampli_phase_err_thermal
                                                    (df_0_2, df_1_2, df_2_2, df_3_2, df_4_2, df_5_2,
                                                     w_pdfs.sine_with_phase_20, param_names, 1))
        amplitudes_3, phases_3, err_a_3, err_p_3 = (get_ampli_phase_err_thermal
                                                    (df_0_3, df_1_3, df_2_3, df_3_3, df_4_3, df_5_3,
                                                     w_pdfs.sine_with_phase_30, param_names, 2))
        amplitudes_4, phases_4, err_a_4, err_p_4 = (get_ampli_phase_err_thermal
                                                    (df_0_4, df_1_4, df_2_4, df_3_4, df_4_4, df_5_4,
                                                     w_pdfs.sine_with_phase_60, param_names, 3))

        phases_1 = phases_1 - phases_1[0]
        phases_1[4] = phases_1[4] + np.pi
        phases_1[5] = phases_1[5] + np.pi
        phases_2 = phases_2 - phases_2[0]
        phases_3 = phases_3 - phases_3[0]
        phases_4 = phases_4 - phases_4[0]

        package_0 = [amplitudes_1, phases_1, err_a_1, err_p_1]
        package_1 = [amplitudes_2, phases_2, err_a_2, err_p_2]
        package_2 = [amplitudes_3, phases_3, err_a_3, err_p_3]
        package_3 = [amplitudes_4, phases_4, err_a_4, err_p_4]
        packages = {"package_0": package_0,
                    "package_1": package_1,
                    "package_2": package_2,
                    "package_3": package_3}

        run_thermal_plots(packages)
    else:
        in_phase_data = data_management.send_data_electrical("In_Phase")
        out_phase_data = data_management.send_data_electrical("Out_of_Phase")

        # Two separate dictionaries holding results per frequency
        in_phase_results = {}
        out_phase_results = {}

        for id, df in in_phase_data.items():
            func_name = f"sine_with_phase_elec_{id}"
            chosen_pdf = getattr(w_pdfs, func_name)

            amplitudes, phases, err_a, err_p, results = get_ampli_phase_err_electrical(
                df, chosen_pdf, param_names, id, True
            )

            # w_outputer.plot_fitted_electrical_waves(df, chosen_pdf, results, id, True)

            in_phase_results[id] = {
                "A_in": amplitudes[0],
                "A_out": amplitudes[1],
                "phi_in": phases[0],
                "phi_out": phases[1],
                "err_A_in": err_a[0],
                "err_A_out": err_a[1],
                "err_phi_in": err_p[0],
                "err_phi_out": err_p[1],
            }

        for id, df in out_phase_data.items():
            func_name = f"sine_with_phase_elec_{id}"
            chosen_pdf = getattr(w_pdfs, func_name)

            amplitudes, phases, err_a, err_p, results = get_ampli_phase_err_electrical(
                df, chosen_pdf, param_names, id, False
            )

            # w_outputer.plot_fitted_electrical_waves(df, chosen_pdf, results, id, False)

            out_phase_results[id] = {
                "A_in": amplitudes[0],
                "A_out": amplitudes[1],
                "phi_in": phases[0],
                "phi_out": phases[1],
                "err_A_in": err_a[0],
                "err_A_out": err_a[1],
                "err_phi_in": err_p[0],
                "err_phi_out": err_p[1],
            }

        plot_dispersion_relation(in_phase_results, out_phase_results)

def compute_k_err(phi_in, phi_out, err_phi_in, err_phi_out, L):
    eps = np.sqrt(err_phi_in ** 2 + err_phi_out ** 2)

    # Construct phase difference modulo 2π
    dphi = (phi_out - phi_in) % (2 * np.pi)

    # Check closeness to 0 or π (two resonance classifications)
    dist_to_0 = abs(dphi - 0)
    dist_to_pi = abs(dphi - np.pi)

    ambiguous = (dist_to_0 < eps) or (dist_to_pi < eps)

    if ambiguous:
        return np.pi / L  # Δn = ±1
    else:
        return 0.0

def plot_dispersion_relation(in_phase_results, out_phase_results):
    in_freqs = np.array(sorted(int(fid) * 10 for fid in in_phase_results.keys()))
    out_freqs = np.array(sorted(int(fid) * 10 for fid in out_phase_results.keys()))
    in_modes = np.array([2 * i + 1 for i in range(len(in_freqs))])  # odd
    out_modes = np.array([2 * i for i in range(len(out_freqs))])  # even
    L = 40
    w_in = 2 * np.pi * in_freqs
    w_out = 2 * np.pi * out_freqs
    k_in = (in_modes + 1) * np.pi / L
    k_out = (out_modes + 1) * np.pi / L

    out_freq_errors = np.array([0.042e3, 0.08e3, 0.08e3, 0.09e3, 0.10e3])
    in_freq_errors  = np.array([0.06e3, 0.08e3, 0.08e3, 0.09e3])
    err_w_in  = 2 * np.pi * in_freq_errors
    err_w_out = 2 * np.pi * out_freq_errors
    err_k_in = np.zeros_like(k_in)
    err_k_out = np.zeros_like(k_out)

    # The number of frequency entries must match the keys order
    in_keys = sorted(in_phase_results.keys(), key=lambda x: int(x))
    out_keys = sorted(out_phase_results.keys(), key=lambda x: int(x))

    # In-phase modes
    for i, key in enumerate(in_keys):
        phi_in = in_phase_results[key]["phi_in"]
        phi_out = in_phase_results[key]["phi_out"]
        err_phi_in = in_phase_results[key]["err_phi_in"]
        err_phi_out = in_phase_results[key]["err_phi_out"]

        err_k_in[i] = compute_k_err(phi_in, phi_out, err_phi_in, err_phi_out, L)

    # Out-of-phase modes
    for i, key in enumerate(out_keys):
        phi_in = out_phase_results[key]["phi_in"]
        phi_out = out_phase_results[key]["phi_out"]
        err_phi_in = out_phase_results[key]["err_phi_in"]
        err_phi_out = out_phase_results[key]["err_phi_out"]

        err_k_out[i] = compute_k_err(phi_in, phi_out, err_phi_in, err_phi_out, L)

    k_all = np.concatenate([k_in, k_out])
    w_all = np.concatenate([w_in, w_out])
    slope, intercept = np.polyfit(k_all, w_all, 1)
    k_fit = np.linspace(min(k_all), max(k_all), 200)
    w_fit = slope * k_fit + intercept

    print("\n---- Dispersion Fit (ω vs k) ----")
    print(f"Fitted slope:     {slope:.4f} rad/s per (rad/segment)")
    print(f"Fitted intercept: {intercept:.4f}")

    residuals_in = w_in - (slope * k_in + intercept)
    residuals_out = w_out - (slope * k_out + intercept)

    w_outputer.plot_dispersion_plot(k_out, k_in, w_out, w_in, err_w_out, err_w_in, k_fit, w_fit,
                         residuals_out, residuals_in, err_k_out+0.001, err_k_in+0.001)
