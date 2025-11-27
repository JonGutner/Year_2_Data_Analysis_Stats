import numpy as np
import pandas as pd
from pathlib import Path
from Waves_Lab import w_estimators, w_outputer, w_pdfs, data_management

def run_tests(df, i, chosen_pdf, param_names, j):
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
    result = w_estimators.mle_fit(y, sine_nll, init_params=init_params, method="BFGS", is_pdf=False)

    # Plot function including DC offset
    plot_pdf = lambda x, amplitude, phase, offset: chosen_pdf(x, amplitude, phase, offset)

    # Print results
    w_outputer.print_results(t, y, result, i, j, param_names, pdf=chosen_pdf)

    # Show fit using original time array
    w_outputer.show_fit(y, plot_pdf, result["params"], j, t=t, title=f"Therm_{i}")

    return result["params"], result["fisher_errors"]

def run_waves_plots(packages):
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

def get_ampli_phase_err(df_0, df_1, df_2, df_3, df_4, df_5, chosen_pdf, param_names, j):
    amplitudes = []
    phases = []
    err_a = []
    err_p = []

    results, errs = run_tests(df_0, 0, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests(df_1, 1, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests(df_2, 2, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])
    results, errs = run_tests(df_3, 3, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    #EXTRA THERMISTORS:
    results, errs = run_tests(df_4, 4, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    results, errs = run_tests(df_5, 5, chosen_pdf, param_names, j)
    amplitudes.append(np.abs(results[0]))
    phases.append(results[1])
    err_a.append(errs[0])
    err_p.append(errs[1])

    return amplitudes, phases, err_a, err_p

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
        amplitudes_1, phases_1, err_a_1, err_p_1 = (get_ampli_phase_err
                                                    (df_0_1, df_1_1, df_2_1, df_3_1, df_4_1, df_5_1,
                                                     w_pdfs.sine_with_phase_15, param_names, 0))
        amplitudes_2, phases_2, err_a_2, err_p_2 = (get_ampli_phase_err
                                                    (df_0_2, df_1_2, df_2_2, df_3_2, df_4_2, df_5_2,
                                                     w_pdfs.sine_with_phase_20, param_names, 1))
        amplitudes_3, phases_3, err_a_3, err_p_3 = (get_ampli_phase_err
                                                    (df_0_3, df_1_3, df_2_3, df_3_3, df_4_3, df_5_3,
                                                     w_pdfs.sine_with_phase_30, param_names, 2))
        amplitudes_4, phases_4, err_a_4, err_p_4 = (get_ampli_phase_err
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

        run_waves_plots(packages)

    else:
        in_phase_data = data_management.send_data_electrical("In_Phase")
        out_phase_data = data_management.send_data_electrical("Out_of_Phase")

