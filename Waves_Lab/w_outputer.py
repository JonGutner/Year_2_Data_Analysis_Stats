import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import ticker

from Waves_Lab import w_estimators

def print_results(x, y, result, i, j, periods, thermal = True, param_names=None, pdf=None):
    if thermal:
        print(f"\nResults for Thermistor {i} & Period {periods[j]} s:")
    else:
        print(f"\nResults for Frequency {float(i) / 100}kHz & Ch{j + 1}")
    for i, val in enumerate(result["params"]):
        name_str = param_names[i] if param_names else f"param{i}"
        fisher_err = result["fisher_errors"][i]
        low, high = result["profile_intervals"][i]
        print(f"  {name_str}: {val:.4f} ± {fisher_err:.4f} "
              f"(68% CL: {low:.4f} – {high:.4f})")
    print("  -logL:", result["neg_logL"])
    print("  Converged:", result["success"])

    gof = w_estimators.goodness_of_fit_regression(x, y, pdf, result["params"])
    print(f"  Chi2/dof = {gof['chi2']:.2f}/{gof['dof']} "
        f"(p = {gof['p_value']:.3f})")
# ----------------------------
# Updated show_fit supporting DC offset
# ----------------------------
def show_fit(data, pdf, params, t=None, bins=50, title="MLE Fit", save=True):
    """
    Plot histogram + fitted PDF/model.
    For sine waves with DC offset, uses provided t array.
    """

    fig, ax = plt.subplots()

    # Determine x points
    if t is None:
        lo, hi = np.percentile(data, [1, 99])
        if lo == hi:
            lo, hi = np.min(data), np.max(data)
            if lo == hi:
                lo -= 1.0
                hi += 1.0
        x = np.linspace(lo, hi, 5000)
    else:
        x = t

    # Evaluate model
    y_model = pdf(x, *params)

    # Plot data
    ax.scatter(t if t is not None else x, data, s = 3, color='blue', alpha=0.5, label="Data")

    # Plot fitted model
    ax.plot(x, y_model, "r-", label="Fitted Model")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")

    ax.set_title(title)
    ax.legend()

    # Save plot
    if save:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_dir = os.path.join(desktop, "StatsPlots")
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    return fig
#spacing, packages, y_models_a, y_models_p
def show_thermistor_param(spacing, packages, y_m_a_old, y_m_p_old, y_m_a_new, y_m_p_new, periods, save=True, show=False):
    colors = ["red", "blue", "green", "orange", "cyan", "purple", "gold", "teal", "olive", "lime", "crimson"]

    # -----------------------------
    # Amplitude plot
    # -----------------------------
    fig, ax = plt.subplots()
    title = "Amplitude Graph for Thermistors 0-7"

    for i in range(len(periods)):
        package = packages[f"package_{periods[i]}"]
        amplitude = np.ravel(package[0])
        err_a = np.ravel(package[2])
        y_m_a_old_i = np.ravel(y_m_a_old[i])
        y_m_a_new_i = np.ravel(y_m_a_new[i])

        ax.errorbar(spacing, amplitude, yerr=err_a, c=colors[i], linestyle='None',
                    fmt='none', capsize=4, label=f"Periods {periods[i]}s")

        # Plot fitted models
        ax.plot(spacing, y_m_a_old_i, c=colors[i])
        ax.plot(spacing, y_m_a_new_i, c=colors[i], linestyle='--')

    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel("Spacing of Thermistors (m)")
    ax.set_ylabel("Amplitude (log) (°C)")
    ax.legend(fontsize=8)

    # Save plot
    if save:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_dir = os.path.join(desktop, "StatsPlots")
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    if show:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    title = "Phase Graph for Thermistors 0-7"

    for i in range(len(periods)):
        package = packages[f"package_{periods[i]}"]
        phase = np.ravel(package[1])
        err_p = np.ravel(package[3])
        y_m_p_old_i = np.ravel(y_m_p_old[i])
        y_m_p_new_i = np.ravel(y_m_p_new[i])

        ax.errorbar(spacing, phase, yerr=err_p, c=colors[i], linestyle='None',
                    fmt='none', capsize=4, label=f"Periods {periods[i]}s")

        # Plot fitted models
        ax.plot(spacing, y_m_p_old_i, c=colors[i])
        ax.plot(spacing, y_m_p_new_i, c=colors[i], linestyle='--')

    ax.set_title(title)
    ax.set_xlabel("Spacing of Thermistors (m)")
    ax.set_ylabel("Phase Difference (rad)")
    ax.legend(fontsize=8)

    # Save plot
    if save:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_dir = os.path.join(desktop, "StatsPlots")
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    if show:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    title = "Residual Amplitude Graph for Thermistors 0-7 of the No Heat Loss Model"
    ax.plot(spacing, (y_m_a_new_i - y_m_a_new_i), c="grey")

    for i in range(len(periods)):
        package = packages[f"package_{periods[i]}"]
        amplitude = np.ravel(package[0])
        err_a = np.ravel(package[2])
        y_m_a_old_i = np.ravel(y_m_a_old[i])

        ax.errorbar(spacing+i*0.0003, amplitude-y_m_a_old_i, yerr=err_a, c=colors[i], linestyle='None',
                    fmt='none', capsize=4, label=f"Periods {periods[i]}s")

    ax.set_title(title)
    ax.set_xlabel("Spacing of Thermistors (m)")
    ax.set_ylabel("Residual: Amplitude - Model Fit (°C)")
    ax.legend(fontsize=8)

    # Save plot
    if save:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_dir = os.path.join(desktop, "StatsPlots")
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    if show:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    title = "Residual Amplitude Graph for Thermistors 0-7 of the Newton's Law of Cooling Model"
    ax.plot(spacing, (y_m_a_new_i - y_m_a_new_i), c="grey")

    for i in range(len(periods)):
        package = packages[f"package_{periods[i]}"]
        amplitude = np.ravel(package[0])
        err_a = np.ravel(package[2])
        y_m_a_new_i = np.ravel(y_m_a_new[i])

        ax.errorbar(spacing+i*0.0003, amplitude-y_m_a_new_i, yerr=err_a, c=colors[i], linestyle='None',
                    fmt='none', capsize=4, label=f"Periods {periods[i]}s")
    ax.set_title(title)
    ax.set_xlabel("Spacing of Thermistors (m)")
    ax.set_ylabel("Residual: Amplitude - Model Fit (°C)")
    ax.legend(fontsize=8)

    # Save plot
    if save:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_dir = os.path.join(desktop, "StatsPlots")
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    if show:
        plt.show()

    plt.close(fig)

def find_diffusivity(packages, periods):
    d = 0.005
    spacing = np.array([0*d, 1*d, 2*d, 3*d, 4*d, 5*d, 6*d, 7*d])
    spacing = spacing + 0.003

    for i in range(len(periods)):
        package = packages[f"package_{periods[i]}"]
        amplitude = package[0]
        phase = package[1]
        freq = 2*np.pi/periods[i]

        diffusivity_a = []
        diffusivity_p = []

        for j in range(1, 6):
            delta_a = np.log(amplitude[j] / amplitude[0])
            delta_p = phase[j] - phase[0]

            diffusivity_a.append((freq * (spacing[j]) ** 2) / (2 * (delta_a ** 2)))
            diffusivity_p.append((freq * (spacing[j]) ** 2) / (2 * (delta_p ** 2)))

        diffusivity_a = np.array(diffusivity_a)
        diffusivity_p = np.array(diffusivity_p)

        # Print diffusivity results
        print(f"\nCalculated diffusivity values OLD MODEL at period {periods[i]}:")
        print(f"From amplitude fits: {diffusivity_a}")
        print(f"From phase fits: {diffusivity_p}")
        print("----------------------")

        # find_diffusivity_pairwise(amplitude, phase, periods, spacing)

def find_diffusivity_pairwise(amplitudes_list, phases_list, periods, spacing):
    """
    Calculate diffusivity from amplitude and phase data using all pairs of spacings.

    Parameters:
    -----------
    amplitudes_list : list of np.array
        Each element is an array of amplitudes [A0, A1, ...] for one period.
    phases_list : list of np.array
        Each element is an array of phases [phi0, phi1, ...] for one period.
    periods : np.array
        Array of periods (seconds) corresponding to each dataset.
    spacing : np.array
        Array of spacing values (meters), same length as amplitudes/phases.

    Returns:
    --------
    D_amp_avg : np.array
        Average diffusivity from amplitude for each period.
    D_phase_avg : np.array
        Average diffusivity from phase for each period.
    """
    D_amp_avg = []
    D_phase_avg = []

    for i in range(len(periods)):
        freq = 2 * np.pi / periods[i]

        D_amp_pairs = []
        D_phase_pairs = []

        n = len(spacing)
        for j in range(n):
            for k in range(j+1, n):
                delta_x = spacing[k] - spacing[j]

                # Amplitude-based diffusivity
                delta_ln = np.log(amplitudes_list[k] / amplitudes_list[j])
                if delta_ln != 0:
                    D_amp_pairs.append(freq * delta_x**2 / (2 * delta_ln**2))

                # Phase-based diffusivity
                delta_phi = phases_list[k] - phases_list[j]
                if delta_phi != 0:
                    D_phase_pairs.append(freq * delta_x**2 / (2 * delta_phi**2))

        # Convert to arrays
        D_amp_pairs = np.array(D_amp_pairs)
        D_phase_pairs = np.array(D_phase_pairs)

        # Average over all pairs
        D_amp_avg.append(np.mean(D_amp_pairs))
        D_phase_avg.append(np.mean(D_phase_pairs))

        # Print results
        print(f"\nPeriod: {periods[i]} s")
        print(f"Amplitude-based diffusivity (all pairs): {D_amp_pairs}")
        print(f"Averaged D from amplitude: {np.mean(D_amp_pairs):.6e}")
        print(f"Phase-based diffusivity (all pairs): {D_phase_pairs}")
        print(f"Averaged D from phase: {np.mean(D_phase_pairs):.6e}")

    return np.array(D_amp_avg), np.array(D_phase_avg)

def plot_initial_electrical_plots(data_frames):
    for id, df in data_frames.items():
        plt.scatter(df["time"], df["ch1"], label="Ch1", color="red", s=1)
        plt.scatter(df["time"], df["ch2"], label="Ch2", color="green", s=1)
        plt.scatter(df["time"], df["ch3"], label="Ch3", color="blue", s=1)

        plt.legend()
        plt.title(f"{float(id) / 100}kHz")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.show()

def plot_fitted_electrical_waves(df, chosen_pdf, results, id, in_phase=True):
    time = df['time']
    plt.scatter(time, df['ch1'], s=1, color='red', label="Ch1")
    plt.scatter(time, df['ch2'], s=1, color='green', label="Ch2")
    plt.scatter(time, df['ch3'], s=1, color='orange', label="Ch3")

    plot_pdf = lambda x, amplitude, phase, offset: chosen_pdf(x, amplitude, phase, offset)
    plt.plot(time, plot_pdf(time, *results[0]), color="black", linestyle="--", )
    plt.plot(time, plot_pdf(time, *results[1]), color="black", linestyle="--", )
    plt.plot(time, plot_pdf(time, *results[2]), color="black", linestyle="--", )

    plt.legend()
    if in_phase:
        plt.title(f"Fitted Electrical Sines in Phase at Frequency {int(id)*10}Hz")
    else:
        plt.title(f"Fitted Electrical Sines out of Phase at Frequency {int(id)*10}Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.show()

def plot_dispersion_plot(k_out, k_in, w_out, w_in, err_w_out, err_w_in, k_fit, w_fit,
                         residuals_out, residuals_in, err_k_out, err_k_in):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.set_title("Dispersion Relation (Task 2.7a)")
    ax.set_xlabel("Wavenumber k (rad/segment)")
    ax.set_ylabel("Angular frequency ω (rad/s)")
    ax.errorbar(k_out, w_out, yerr=err_w_out, xerr=err_k_out, fmt='o', capsize=5, ecolor='blue', label="Out-of-phase")
    ax.errorbar(k_in, w_in, yerr=err_w_in, xerr=err_k_in, fmt='o', capsize=5, ecolor='red', label="In-phase")
    ax.plot(k_fit, w_fit, 'k--', label="Linear fit")
    ax.grid(True)
    ax.legend()

    ax_r = axs[1]
    ax_r.set_title("Fit residuals")
    ax_r.set_xlabel("Wavenumber k (rad/segment)")
    ax_r.set_ylabel("Residual  Δω = ω_measured − ω_fit")
    ax_r.errorbar(k_out, residuals_out, yerr=err_w_out, xerr=err_k_out, fmt='o', capsize=5, ecolor='blue', label="Out-of-phase")
    ax_r.errorbar(k_in, residuals_in, yerr=err_w_in, xerr=err_k_in, fmt='o', capsize=5, ecolor='red', label="In-phase")
    ax_r.axhline(0, color='gray', linestyle='--')
    ax_r.grid(True)
    ax_r.legend()
    plt.tight_layout()
    plt.show()