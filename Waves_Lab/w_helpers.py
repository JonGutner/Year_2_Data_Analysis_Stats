import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from Waves_Lab import w_estimators, w_outputer, w_pdfs, data_management

def run_tests(df, i, chosen_pdf, param_names, j, init_params,
              periods = None, close_fig=False, thermal=True):
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

    if thermal:
        # Thermal initial guess
        init_params = [0.5 * (np.max(y) + np.min(y)), np.pi / 2, np.mean(y)]

        result = w_estimators.mle_fit(
            y, sine_nll, init_params=init_params,
            method="BFGS", is_pdf=False
        )

        plot_pdf = lambda x, amplitude, phase, offset: chosen_pdf(
            x, amplitude, phase, offset)

        w_outputer.print_results(
            t, y, result, i, j, periods,True, param_names, pdf=chosen_pdf
        )

        fig = w_outputer.show_fit(
            y, plot_pdf, result["params"], t=t,
            title=f"Therm_{i}_period_{periods[j]}"
        )
    else:
        # Electrical: use provided init_params
        result = w_estimators.mle_fit(
            y, sine_nll, init_params=init_params,
            method="BFGS", is_pdf=False
        )

        w_outputer.print_results(
            t, y, result, i, j, periods,False, param_names, pdf=chosen_pdf
        )

    if close_fig and fig is not None:
        plt.close(fig)

    return result["params"], result["fisher_errors"]

# ======================================================================
#  THERMAL PART (unchanged)
# ======================================================================

def run_thermal_plots(packages, periods):
    y_models_a_old = []
    y_models_a_new = []
    y_models_p_old = []
    y_models_p_new = []
    popts_a = []
    pcovs_a = []
    popts_p = []
    pcovs_p = []

    d = 0.005
    spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * d + 0.003

    for period in periods:
        package = packages[f"package_{period}"]
        amplitude = np.ravel(package[0])  # ensure 1D
        phase = np.ravel(package[1])  # ensure 1D

        # -------------------------
        # Old models (keep as-is)
        # -------------------------
        params_a_old = [amplitude[0], -1]
        popt_a_old, pcov_a_old = w_estimators.amplitude_fit(spacing, amplitude, params_a_old)
        y_model_a_old = np.ravel(w_pdfs.amplitude_waves(spacing, *popt_a_old))

        sigma_old = np.asarray(package[2], dtype=float)  # same length as amplitude
        gof_old = w_estimators.goodness_of_fit_regression(
            spacing,
            np.asarray(amplitude, dtype=float),
            w_pdfs.amplitude_waves,
            popt_a_old,
            sigma=sigma_old
        )
        print("=========OLD=MODEL=START==========")
        print(f"[Period {period}s] Old amplitude model: "
              f"χ² = {gof_old['chi2']:.2f}, dof = {gof_old['dof']}, "
              f"χ²/dof = {gof_old['chi2'] / gof_old['dof']:.2f}, "
              f"p = {gof_old['p_value']:.3f}")
        print("=========OLD=MODEL=END==========")

        params_p_old = [0.3, 0]
        popt_p_old, pcov_p_old = w_estimators.phase_fit(spacing, phase, params_p_old)
        y_model_p_old = np.ravel(w_pdfs.phase_waves(spacing, *popt_p_old))
        #
        # sigma_phase = np.asarray(package[3], dtype=float)  # same length as phase
        # gof_old = w_estimators.goodness_of_fit_regression(
        #     spacing,
        #     np.asarray(phase, dtype=float),
        #     w_pdfs.phase_waves,
        #     popt_p_old,
        #     sigma=sigma_phase
        # )
        # print("===================")
        # print(f"[Period {period}s] Old phase model: "
        #       f"χ² = {gof_old['chi2']:.2f}, dof = {gof_old['dof']}, "
        #       f"χ²/dof = {gof_old['chi2'] / gof_old['dof']:.2f}, "
        #       f"p = {gof_old['p_value']:.3f}")
        # print("===================")

        # -------------------------
        # New Robin-boundary model
        # -------------------------
        omega = 2 * np.pi / period
        L = 0.043  # cylinder length in meters

        def robin_amp_phase_model(x, Cpr, Cpi, D, h_over_k):
            """
            Return complex amplitude, amplitude, and raw phase (wrapped).
            """
            X = w_pdfs.X_model(x, Cpr, Cpi, D, h_over_k, omega, L)
            amp = np.abs(X)
            phi = np.angle(X)  # in (-π, π]
            return amp, phi

        def robin_amp_model(x, Cpr, Cpi, D, h_over_k):
            omega = 2 * np.pi / period
            L = 0.043
            amp, _ = robin_amp_phase_model(x, Cpr, Cpi, D, h_over_k)
            return amp

        def robin_phase_model_unwrapped(x, Cpr, Cpi, D, h_over_k):
            """
            Phase model processed exactly like the data:
            relative to thermistor 0 + np.unwrap + optional positive shift.
            """
            # compute at all sensor positions (spacing)
            _, phi_raw = robin_amp_phase_model(spacing, Cpr, Cpi, D, h_over_k)

            # relative to thermistor 0
            phi_rel = phi_raw - phi_raw[0]

            # unwrap
            phi_unw = np.unwrap(phi_rel)

            # OPTIONAL: same positive shift convention as data
            if phi_unw[1] < 0:
                phi_unw += 2 * np.pi

            # Now select the values corresponding to x that curve_fit passes.
            # In your usage, x is exactly 'spacing', so we can just return phi_unw.
            # To be safe (if order always matches):
            return phi_unw

        # initial guesses and bounds
        p0 = [amplitude[0], 0.0, 1e-6, 20.0]  # Cpr, Cpi, D, h_over_k
        bounds = ([-np.inf, -np.inf, 1e-8, 0.0], [np.inf, np.inf, 1e-3, 1e4])

        # fit amplitude
        popt_a_new, pcov_a_new = curve_fit(robin_amp_model, spacing, amplitude, p0=p0, bounds=bounds)
        y_model_a_new = np.ravel(robin_amp_model(spacing, *popt_a_new))

        sigma_new = np.asarray(package[2], dtype=float)  # same length as amplitude
        gof_new = w_estimators.goodness_of_fit_regression(
            spacing,
            np.asarray(amplitude, dtype=float),
            robin_amp_model,
            popt_a_new,
            sigma=sigma_new
        )
        print("=========NEW=MODEL=START==========")
        print(f"[Period {period}s] New amplitude model: "
              f"χ² = {gof_new['chi2']:.2f}, dof = {gof_new['dof']}, "
              f"χ²/dof = {gof_new['chi2'] / gof_new['dof']:.2f}, "
              f"p = {gof_new['p_value']:.3f}")
        print("=========NEW=MODEL=END==========")

        # fit phase
        popt_p_new, pcov_p_new = curve_fit(
            robin_phase_model_unwrapped,
            spacing,
            phase,  # 'phase' is already unwrapped from the package
            p0=p0,
            bounds=bounds
        )

        # evaluate model phase for plotting (already unwrapped and shifted)
        y_model_p_new = np.ravel(robin_phase_model_unwrapped(spacing, *popt_p_new))

        # -------------------------
        # Store results
        # -------------------------
        popts_a.append(popt_a_old)
        pcovs_a.append(pcov_a_old)
        popts_p.append(popt_p_old)
        pcovs_p.append(pcov_p_old)

        y_models_a_old.append(y_model_a_old)
        y_models_a_new.append(y_model_a_new)
        y_models_p_old.append(y_model_p_old)
        y_models_p_new.append(y_model_p_new)

    w_outputer.show_thermistor_param(spacing, packages, y_models_a_old, y_models_p_old,
                                     y_models_a_new, y_models_p_new, periods)
    w_outputer.find_diffusivity(popts_a, pcovs_a, popts_p, pcovs_p, packages, periods) #TODO: HERE

def get_ampli_phase_err_thermal(df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7,
                                chosen_pdf, param_names, j, periods):
    amplitudes = []
    phases = []
    err_a = []
    err_p = []

    for i, df in enumerate([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7]):
        results, errs = run_tests(df, i, chosen_pdf, param_names, j, None, periods,True)
        amplitudes.append(np.abs(results[0]))
        phases.append(results[1])
        err_a.append(errs[0])
        err_p.append(errs[1])

    return amplitudes, phases, err_a, err_p

# ======================================================================
#  ELECTRICAL: GET AMPLITUDES/PHASES FOR EACH FREQUENCY
# ======================================================================

def get_ampli_phase_err_electrical(df, chosen_pdf, param_names, freq,
                                   is_in_phase=True):
    """
    Fit sine-with-phase-with-DC to channels ch1, ch2, ch3 in df.
    Return amplitude, phase, errors for each channel.
    """
    amplitudes = []
    phases = []
    err_a = []
    err_p = []
    results_list = []

    df_1 = pd.DataFrame({'time': df['time'], 'ch1': df['ch1']})
    df_2 = pd.DataFrame({'time': df['time'], 'ch2': df['ch2']})
    df_3 = pd.DataFrame({'time': df['time'], 'ch3': df['ch3']})

    if is_in_phase:
        init_params = [
            0.5 * (np.max(df['ch1']) - np.min(df['ch1'])),
            0.0,
            np.mean(df['ch1'])
        ]
    else:
        init_params = [
            0.5 * (np.max(df['ch1']) - np.min(df['ch1'])),
            np.pi,
            np.mean(df['ch1'])
        ]

    for j, dfi in enumerate([df_1, df_2, df_3]):
        results, errs = run_tests(
            dfi, freq, chosen_pdf, param_names, j,
            init_params, False, False
        )
        amplitudes.append(np.abs(results[0]))
        phases.append(results[1])
        err_a.append(errs[0])
        err_p.append(errs[1])
        results_list.append(results)

    return amplitudes, phases, err_a, err_p, results_list

# ======================================================================
#  TOP-LEVEL LOADER: THERMAL vs ELECTRICAL
# ======================================================================

def load_run_thermal_vs_electrical(param_names, periods, is_thermal=False):
    if is_thermal:
        # -----------------------------
        # THERMAL experiment
        # -----------------------------
        packages = {}

        for i in range(len(periods)):
            data_name = f"period_{periods[i]}.csv"
            df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7 = \
                data_management.send_data_thermal(data_name)

            func_pdf = getattr(w_pdfs, f"sine_with_phase_{periods[i]}")

            amplitudes, phases, err_a, err_p = get_ampli_phase_err_thermal(
                df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7,
                func_pdf, param_names, i, periods)

            phases = np.asarray(phases, dtype=float)

            # Phase difference relative to thermistor 0
            phase_diff = phases - phases[0]

            # SIMPLE spatial unwrapping: remove only ±2π jumps
            phase_diff_unwrapped = np.unwrap(phase_diff)

            # OPTIONAL: shift the whole curve to be mostly positive
            if phase_diff_unwrapped[1] < 0:
                phase_diff_unwrapped += 2 * np.pi

            package = [amplitudes, phase_diff_unwrapped, err_a, err_p]
            packages[f"package_{periods[i]}"] = package

        run_thermal_plots(packages, periods)

    else:
        # -----------------------------
        # ELECTRICAL experiment
        # -----------------------------
        in_phase_data = data_management.send_data_electrical("In_Phase")
        out_phase_data = data_management.send_data_electrical("Out_of_Phase")

        in_phase_results = {}
        out_phase_results = {}

        # In-phase frequencies
        for id, df in in_phase_data.items():
            func_name = f"sine_with_phase_elec_{id}"
            chosen_pdf = getattr(w_pdfs, func_name)
            amplitudes, phases, err_a, err_p, results = \
                get_ampli_phase_err_electrical(df, chosen_pdf,
                                               param_names, id, True)

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

        # Out-of-phase frequencies
        for id, df in out_phase_data.items():
            func_name = f"sine_with_phase_elec_{id}"
            chosen_pdf = getattr(w_pdfs, func_name)
            amplitudes, phases, err_a, err_p, results = \
                get_ampli_phase_err_electrical(df, chosen_pdf,
                                               param_names, id, False)

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

        # 2.7(a): dispersion (unchanged)
        k_w_data = plot_dispersion_relation(in_phase_results,
                                            out_phase_results)

        # 2.7(c) & (d): velocities and cutoff (now built from 2.7a k, ω)
        results = run_all(in_phase_results, out_phase_results,
                          k_w_data, do_plots=True)

# ======================================================================
#  ELECTRICAL: DISPERSION (2.7a)  — UNCHANGED, BUT RETURNS k & ω
# ======================================================================

def compute_k_err(phi_in, phi_out, err_phi_in, err_phi_out, L):
    eps = np.sqrt(err_phi_in ** 2 + err_phi_out ** 2)

    # Phase difference modulo 2π
    dphi = (phi_out - phi_in) % (2 * np.pi)

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

    out_freq_errors = np.array([0.042e3, 0.08e3, 0.08e3, 0.09e3])
    in_freq_errors = np.array([0.06e3, 0.08e3, 0.08e3, 0.10e3])

    err_w_in = 2 * np.pi * in_freq_errors
    err_w_out = 2 * np.pi * out_freq_errors
    err_k_in = np.zeros_like(k_in)
    err_k_out = np.zeros_like(k_out)

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

    k_all = np.concatenate([k_out, k_in])
    w_all = np.concatenate([w_out, w_in])
    err_w_all = np.concatenate([err_w_out, err_w_in])
    err_k_all = np.concatenate([err_k_out, err_k_in])

    # Sinusoidal dispersion fit: ω(k) = A sin(k/2)
    def omega_model(k, A):
        return A * np.sin(k / 2)

    popt, pcov = curve_fit(
        omega_model, k_all, w_all,
        sigma=err_w_all, absolute_sigma=True,
        p0=[1e5]
    )
    A_fit = popt[0]
    A_err = np.sqrt(pcov[0, 0])

    k_fit = np.linspace(min(k_all), max(k_all), 300)
    w_fit = omega_model(k_fit, A_fit)

    print("\n---- TRUE DISPERSION FIT ----")
    print(f"A (2/sqrt(LC)) = {A_fit:.4e} ± {A_err:.4e} rad/s")
    print(f"Cutoff frequency ω_max = {A_fit:.4e} rad/s")
    print(f"Cutoff f_max = {A_fit / (2 * np.pi):.4f} Hz")

    residuals_out = w_out - omega_model(k_out, A_fit)
    residuals_in = w_in - omega_model(k_in, A_fit)

    w_outputer.plot_dispersion_plot(
        k_out, k_in, w_out, w_in,
        err_w_out, err_w_in,
        k_fit, w_fit,
        residuals_out, residuals_in,
        err_k_out + 0.001, err_k_in + 0.001
    )

    # return all k, ω, and uncertainties for later
    # return all k, ω, and uncertainties for later use (2.7c)
    return {
        "k_in": k_in,
        "k_out": k_out,
        "w_in": w_in,
        "w_out": w_out,
        "in_freqs": in_freqs,
        "out_freqs": out_freqs,
        "err_k_in": err_k_in,
        "err_k_out": err_k_out,
        "err_w_in": err_w_in,
        "err_w_out": err_w_out,
        "A_fit": A_fit,
        "A_err": A_err
    }

# ======================================================================
#  2.7(d) HELPER — CUTOFF FROM AMPLITUDE
# ======================================================================

def cutoff_from_amplitude(freq, amp_ratio, sigma_amp, n_plateau=3):
    """
    Given freq (Hz) and amplitude ratio A_out/A_in (linear) with uncertainties,
    compute a -3 dB cutoff relative to the low-frequency plateau.

    Returns (cutoff_freq, cutoff_unc, plateau_dB, target_dB).
    If the data do not cross the -3 dB level, cutoff_freq and cutoff_unc are NaN.
    """
    freq = np.asarray(freq)
    amp_ratio = np.asarray(amp_ratio)
    sigma_amp = np.asarray(sigma_amp)

    # Convert to dB
    amp_dB = 20.0 * np.log10(amp_ratio)
    sigma_amp_dB = (20.0 / np.log(10.0)) * (sigma_amp / amp_ratio)

    # Plateau and target
    n_plateau = min(n_plateau, len(amp_dB))
    plateau_dB = np.mean(amp_dB[:n_plateau])
    target_dB = plateau_dB - 3.0

    cutoff_freq = np.nan
    cutoff_unc = np.nan

    # Need data on both sides of target_dB
    if np.any(amp_dB > target_dB) and np.any(amp_dB < target_dB):
        diff_sign = amp_dB - target_dB
        idx_candidates = np.where(diff_sign[:-1] * diff_sign[1:] <= 0)[0]
        if len(idx_candidates) > 0:
            idx = idx_candidates[0]
            f1, f2 = freq[idx], freq[idx + 1]
            y1, y2 = amp_dB[idx], amp_dB[idx + 1]
            if y2 != y1:
                t = (target_dB - y1) / (y2 - y1)
                cutoff_freq = f1 + t * (f2 - f1)
                cutoff_unc = np.abs(f2 - f1) * np.sqrt(
                    sigma_amp_dB[idx] ** 2 + sigma_amp_dB[idx + 1] ** 2
                ) / np.abs(y2 - y1)

    return cutoff_freq, cutoff_unc, plateau_dB, target_dB, amp_dB, sigma_amp_dB

def transfer_model(freq, fc, n):
    """
    Simple magnitude model for amplitude ratio A_out/A_in (linear):

        |H(f)| = 1 / sqrt(1 + (f/fc)^n).

    freq : Hz
    fc   : cutoff frequency (Hz)
    n    : effective slope parameter (unitless)

    This is heuristic but captures "flat then rolling off" behaviour.
    """
    x = freq / fc
    return 1.0 / np.sqrt(1.0 + x**n)

def fit_transfer_function(freq, amp_ratio, sigma_amp=None):
    """
    Fit the amplitude-vs-frequency data to the transfer_model.

    Parameters
    ----------
    freq      : array of frequencies (Hz)
    amp_ratio : array of amplitude ratios A_out/A_in (linear)
    sigma_amp : uncertainties in amp_ratio (linear). If None, equal weights.

    Returns
    -------
    fc_fit, fc_err, n_fit, n_err
    """
    freq = np.asarray(freq, dtype=float)
    amp_ratio = np.asarray(amp_ratio, dtype=float)

    if sigma_amp is None:
        sigma = None
    else:
        sigma = np.asarray(sigma_amp, dtype=float)

    # Initial guesses:
    # fc ~ 1e5 Hz (from theory), n ~ 2
    p0 = [1.5e5, 2.0]

    try:
        popt, pcov = curve_fit(
            transfer_model, freq, amp_ratio,
            p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000
        )
        fc_fit, n_fit = popt
        fc_err = np.sqrt(pcov[0, 0]) if pcov.shape[0] > 0 else np.nan
        n_err  = np.sqrt(pcov[1, 1]) if pcov.shape[0] > 1 else np.nan
    except Exception as e:
        print("Warning: transfer function fit failed:", e)
        fc_fit, fc_err, n_fit, n_err = np.nan, np.nan, np.nan, np.nan

    return fc_fit, fc_err, n_fit, n_err

# ======================================================================
#  NEW run_all: USE 2.7(a) K, Ω FOR 2.7(c) + AMPLITUDE FOR 2.7(d)
# ======================================================================

def run_all(in_phase_results, out_phase_results, k_w_data, do_plots=True):
    """
    High-level function for Tasks 2.7(c) and 2.7(d).

    Inputs
    ------
    in_phase_results, out_phase_results:
        dicts keyed by id string ('1','2',...) with
        A_in, A_out, phi_in, phi_out, err_A_in, err_A_out, err_phi_in, err_phi_out.

    k_w_data:
        dict returned by plot_dispersion_relation containing
        k_in, k_out, w_in, w_out, in_freqs, out_freqs, err_k_in, err_k_out,...

    do_plots: bool
        If True, plot 2.7(c) and 2.7(d) figures.

    This function:
      * builds amplitude ratios vs frequency from the dicts,
      * reuses k and ω from 2.7(a) (k_w_data) to compute v_phase & v_group,
      * finds the -3 dB cutoff if the data cross it.
    """

    # ------------------------------------------------------
    # 1. Build amplitude ratio arrays vs frequency
    # ------------------------------------------------------
    def freq_from_id(fid):
        return 10.0 * float(int(fid))  # Hz

    # Use the same ids and ordering for amplitudes as for k_w_data
    # (in_freqs & out_freqs are already sorted by id)
    in_keys = sorted(in_phase_results.keys(), key=lambda x: int(x))
    out_keys = sorted(out_phase_results.keys(), key=lambda x: int(x))

    in_freqs = k_w_data["in_freqs"]
    out_freqs = k_w_data["out_freqs"]

    amp_freq_list = []
    amp_ratio_list = []
    sigma_amp_list = []

    # Out-of-phase amplitudes
    for key in out_keys:
        d = out_phase_results[key]
        A_in = d["A_in"]
        A_out = d["A_out"]
        err_A_in = d["err_A_in"]
        err_A_out = d["err_A_out"]
        if A_in <= 0 or A_out <= 0:
            continue
        amp_freq_list.append(freq_from_id(key))
        amp_ratio_list.append(A_out / A_in)
        sigma_amp_list.append(
            np.sqrt((err_A_out / A_in) ** 2 +
                    ((A_out * err_A_in) / (A_in ** 2)) ** 2)
        )

    # In-phase amplitudes
    for key in in_keys:
        d = in_phase_results[key]
        A_in = d["A_in"]
        A_out = d["A_out"]
        err_A_in = d["err_A_in"]
        err_A_out = d["err_A_out"]
        if A_in <= 0 or A_out <= 0:
            continue
        amp_freq_list.append(freq_from_id(key))
        amp_ratio_list.append(A_out / A_in)
        sigma_amp_list.append(
            np.sqrt((err_A_out / A_in) ** 2 +
                    ((A_out * err_A_in) / (A_in ** 2)) ** 2)
        )

    amp_freq = np.array(amp_freq_list)
    amp_ratio = np.array(amp_ratio_list)
    sigma_amp = np.array(sigma_amp_list)

    # Sort amplitude data by frequency
    sort_idx_amp = np.argsort(amp_freq)
    amp_freq = amp_freq[sort_idx_amp]
    amp_ratio = amp_ratio[sort_idx_amp]
    sigma_amp = sigma_amp[sort_idx_amp]

    # ------------------------------------------------------
    # 2. Build combined k, ω, freq for velocities using 2.7(a) data
    # ------------------------------------------------------
    k_in = k_w_data["k_in"]
    k_out = k_w_data["k_out"]
    w_in = k_w_data["w_in"]
    w_out = k_w_data["w_out"]
    err_k_in = k_w_data["err_k_in"]
    err_k_out = k_w_data["err_k_out"]

    freq_all = np.concatenate([out_freqs, in_freqs])
    k_all = np.concatenate([k_out, k_in])
    w_all = np.concatenate([w_out, w_in])
    err_k_all = np.concatenate([err_k_out, err_k_in])

    # Sort by increasing k (or equivalently by increasing mode)
    sort_idx = np.argsort(k_all)
    k_all = k_all[sort_idx]
    w_all = w_all[sort_idx]
    freq_all = freq_all[sort_idx]
    err_k_all = err_k_all[sort_idx]

    # Phase velocity (segments/s)
    v_phase = w_all / k_all

    # Group velocity by finite difference on ω(k)
    d_omega = np.diff(w_all)
    d_k = np.diff(k_all)
    d_k_safe = np.where(np.abs(d_k) < 1e-12, 1e-12, d_k)
    v_group = d_omega / d_k_safe

    # Midpoints for plotting v_group vs frequency
    freq_mid = 0.5 * (freq_all[:-1] + freq_all[1:])

    # Uncertainty in v_group from σ_k
    sigma_k_all = err_k_all
    sigma_v_group = np.abs(d_omega) * np.sqrt(
        sigma_k_all[:-1] ** 2 + sigma_k_all[1:] ** 2
    ) / (d_k_safe ** 2)

    # ------------------------------------------------------
    # 3. 2.7(d): cutoff from amplitude
    # ------------------------------------------------------
    cutoff_freq, cutoff_unc, plateau_dB, target_dB, \
        amp_dB, sigma_amp_dB = cutoff_from_amplitude(
            amp_freq, amp_ratio, sigma_amp, n_plateau=3
        )

    # ------------------------------------------------------
    # 3b. Fit a simple low-pass model to amplitude vs freq
    # ------------------------------------------------------
    fc_fit, fc_fit_err, n_fit, n_fit_err = fit_transfer_function(
        amp_freq, amp_ratio, sigma_amp
    )

    # ------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------
    if do_plots:
        # ---- 2.7(c): velocities ----
        plt.figure(figsize=(8, 5))

        # Numerical velocities from data
        plt.plot(freq_all, v_phase, "o-", label="v_phase (data) = ω/k")
        plt.plot(freq_mid, v_group, "s--", label="v_group (data) = Δω/Δk")
        plt.fill_between(
            freq_mid,
            v_group - sigma_v_group,
            v_group + sigma_v_group,
            alpha=0.2,
            color="orange",
            label="v_group 1σ (data)",
        )

        # ---- Analytic velocities from fitted dispersion ω(k) = A sin(k/2) ----
        A_fit = k_w_data["A_fit"]

        # k range for smooth theoretical curves (within measured range)
        k_min = np.min(k_all)
        k_max = np.max(k_all)
        k_grid = np.linspace(k_min, k_max, 300)

        # Avoid division by zero for vp_theory near k=0
        # (your k range never includes 0 exactly, but we guard anyway)
        vp_theory = (A_fit * np.sin(k_grid / 2.0)) / np.where(
            np.abs(k_grid) < 1e-12, 1e-12, k_grid
        )
        vg_theory = 0.5 * A_fit * np.cos(k_grid / 2.0)

        # Map k_grid → frequency via the same dispersion model
        omega_grid = A_fit * np.sin(k_grid / 2.0)
        freq_grid = omega_grid / (2.0 * np.pi)

        # Plot theoretical curves
        plt.plot(
            freq_grid,
            vp_theory,
            "k-",
            alpha=0.7,
            label="v_phase (model)",
        )
        plt.plot(
            freq_grid,
            vg_theory,
            "k--",
            alpha=0.7,
            label="v_group (model)",
        )

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Velocity (segments/s)")
        plt.title("Task 2.7(c) — Phase and Group Velocities vs Frequency")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- 2.7(d): amplitude & cutoff ----
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            amp_freq,
            amp_dB,
            yerr=sigma_amp_dB,
            fmt="o",
            capsize=4,
            label="A_out/A_in (data, dB)",
        )

        # Horizontal reference at plateau - 3 dB
        plt.axhline(
            target_dB,
            color="red",
            linestyle="--",
            label=f"{target_dB:.1f} dB (= plateau - 3 dB)",
        )

        # Plot fitted transfer model in dB if fit succeeded
        if np.isfinite(fc_fit):
            f_fit = np.linspace(np.min(amp_freq), np.max(amp_freq), 400)
            amp_fit_lin = transfer_model(f_fit, fc_fit, n_fit)
            amp_fit_dB = 20.0 * np.log10(amp_fit_lin)
            plt.plot(
                f_fit,
                amp_fit_dB,
                "k-",
                alpha=0.7,
                label=f"Model fit (fc ≈ {fc_fit:.1f} Hz)",
            )

        if np.isfinite(cutoff_freq):
            plt.axvline(
                cutoff_freq,
                color="magenta",
                linestyle="--",
                label=f"Data-based fc ≈ {cutoff_freq:.1f} ± {cutoff_unc:.1f} Hz",
            )

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude ratio (dB)")
        plt.title("Task 2.7(d) — Amplitude Ratio vs Frequency")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------
    # 5. Print / return numerical results
    # ------------------------------------------------------
    print("=== Task 2.7(c) results ===")
    print("freq_all (Hz):", freq_all)
    print("k_all (rad/segment):", k_all)
    print("v_phase (segments/s):", v_phase)
    print("v_group midpoints (segments/s):", v_group)
    print("v_group 1σ (segments/s):", sigma_v_group)

    print("\n=== Task 2.7(d) results ===")
    print("Amplitude ratio (linear):", amp_ratio)
    print("Amplitude ratio (dB):", amp_dB)
    if np.isfinite(cutoff_freq):
        print(
            f"Cutoff f_c from -3 dB crossing ≈ {cutoff_freq:.3g} ± {cutoff_unc:.3g} Hz "
            f"(defined at {target_dB:.1f} dB = plateau - 3 dB)"
        )
    else:
        print("No -3 dB crossing in data range: cannot directly measure fc.")

    if np.isfinite(fc_fit):
        print(
            f"Fit to low-pass model: fc_fit ≈ {fc_fit:.3g} ± {fc_fit_err:.3g} Hz, "
            f"n ≈ {n_fit:.2f} ± {n_fit_err:.2f}"
        )
    else:
        print("Transfer function fit failed or not well constrained.")

    return {
        "freq_all": freq_all,
        "k_all": k_all,
        "omega_all": w_all,
        "v_phase": v_phase,
        "v_group": v_group,
        "sigma_v_group": sigma_v_group,
        "freq_mid": freq_mid,
        "amp_freq": amp_freq,
        "amp_ratio": amp_ratio,
        "amp_dB": amp_dB,
        "sigma_amp_dB": sigma_amp_dB,
        "plateau_dB": plateau_dB,
        "target_dB": target_dB,
        "cutoff_freq_data": cutoff_freq,
        "cutoff_unc_data": cutoff_unc,
        "fc_fit_model": fc_fit,
        "fc_fit_err_model": fc_fit_err,
        "n_fit": n_fit,
        "n_fit_err": n_fit_err,
    }