import numpy as np
from Waves_Lab import w_estimators, w_outputer, w_pdfs

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