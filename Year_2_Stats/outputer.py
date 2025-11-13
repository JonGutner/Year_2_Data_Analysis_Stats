#Outputs any results that come of the analysis.
import matplotlib.pyplot as plt
import numpy as np
import os
from Year_2_Stats import estimators, pdfs

def print_results(name, result, param_names=None, data=None, pdf=None):
    print(f"\nResults for {name}:")
    for i, val in enumerate(result["params"]):
        name_str = param_names[i] if param_names else f"param{i}"
        fisher_err = result["fisher_errors"][i]
        low, high = result["profile_intervals"][i]
        print(f"  {name_str}: {val:.4f} ± {fisher_err:.4f} "
              f"(68% CL: {low:.4f} – {high:.4f})")
    print("  -logL:", result["neg_logL"])
    print("  Converged:", result["success"])

    # Goodness-of-fit section
    if data is not None and pdf is not None:
        gof = estimators.goodness_of_fit(data, pdf, result["params"])
        print(f"  Chi2/dof = {gof['chi2']:.2f}/{gof['dof']} "
              f"(p = {gof['p_value']:.3f})")

def print_results_waves(x, y, result, i, j, param_names=None, pdf=None):
    periods = [15, 20, 30, 60]
    print(f"\nResults for Thermistor {i} & Period {periods[j]} s:")
    for i, val in enumerate(result["params"]):
        name_str = param_names[i] if param_names else f"param{i}"
        fisher_err = result["fisher_errors"][i]
        low, high = result["profile_intervals"][i]
        print(f"  {name_str}: {val:.4f} ± {fisher_err:.4f} "
              f"(68% CL: {low:.4f} – {high:.4f})")
    print("  -logL:", result["neg_logL"])
    print("  Converged:", result["success"])

    gof = estimators.goodness_of_fit_regression(x, y, pdf, result["params"])
    print(f"  Chi2/dof = {gof['chi2']:.2f}/{gof['dof']} "
        f"(p = {gof['p_value']:.3f})")

def show_fit_pdf(data, pdf, params, folder_suffix="", bins=50, title="MLE Fit", save=True, show=False):
    """
    Histogram + fitted PDF/PMF overlay.
    - Detects discrete PDFs (Poisson and binomial closures) automatically.
    - For discrete PDFs: plot integer bins and PMF stems.
    - For continuous PDFs: plot histogram + smooth curve.
    """
    fig, ax = plt.subplots()

    # --- Detect discrete PDFs ---
    is_poisson = pdf is pdfs.poisson_pmf
    is_binomial = pdf.__name__ == "pdf" and pdf.__closure__ is not None
    is_discrete = is_poisson or is_binomial

    if is_discrete:
        # Integer data range
        kmin = int(np.floor(np.min(data)))
        kmax = int(np.ceil(np.max(data)))
        k_vals = np.arange(kmin, kmax + 1, 1)

        # Histogram: bin edges centered on integers
        bin_edges = np.arange(kmin - 0.5, kmax + 1.5, 1.0)
        ax.hist(data, bins=bin_edges, density=True, alpha=0.5, label="Data")

        # Fitted PMF
        y = pdf(k_vals, *params)
        markerline, stemlines, baseline = ax.stem(k_vals, y, linefmt="r-", markerfmt="ro", basefmt=" ")
        plt.setp(stemlines, linewidth=1.5)
        plt.setp(markerline, markersize=5)

        ax.set_xlim(kmin - 0.5, kmax + 0.5)

    else:
        # Continuous: clip to 1st–99th percentile for better view
        lo, hi = np.percentile(data, [1.0, 99.0])
        if lo == hi:
            lo, hi = np.min(data), np.max(data)
            if lo == hi:  # constant dataset
                lo -= 1.0
                hi += 1.0

        x = np.linspace(lo, hi, 500)
        y = pdf(x, *params)

        ax.hist(data, bins=bins, range=(lo, hi), density=True, alpha=0.5, label="Data")
        ax.plot(x, y, "r-", label="Fitted PDF")

    ax.set_title(title)
    ax.legend()

    # Save to Desktop/StatsPlots_<suffix>
    if save:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base_dir = "StatsPlots" + (f"_{folder_suffix}" if folder_suffix else "")
        out_dir = os.path.join(desktop, base_dir)
        os.makedirs(out_dir, exist_ok=True)

        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    if show:
        plt.show()

    plt.close(fig)

# ----------------------------
# Updated show_fit supporting DC offset
# ----------------------------
def show_fit_waves(data, pdf, params, j, t=None, bins=50, title="MLE Fit", save=True, show=False):
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

    ax.set_title(title)
    ax.legend()

    # Save plot
    if save:
        periods = [15, 20, 30, 60]
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        out_dir = os.path.join(desktop, "StatsPlots")
        os.makedirs(out_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_title}_num_{periods[j]}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot saved] {out_path}")

    if show:
        plt.show()

    plt.close(fig)
#spacing, packages, y_models_a, y_models_p
def show_thermistor_param(spacing, packages, y_models_a, y_models_p, save=True, show=False):
    #Amplitude
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange']
    frequencies = [15, 20, 30, 60]
    title = "Amplitude Graphs for Thermistors 0-3"

    for i in range(4):
        package = packages[f"package_{i}"]
        amplitude = package[0]
        err_a = package[2]
        y_model_a = y_models_a[i]

        ax.errorbar(spacing, amplitude, yerr=err_a, c=colors[i], linestyle=None,
                    capsize=4, label=f"Frequencies {frequencies[i]}s")

        # Plot fitted model
        ax.plot(spacing, y_model_a, c=colors[i])

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

    if show:
        plt.show()

    plt.close(fig)

    title = "Phase Graphs for Thermistors 0-3"
    fig, ax = plt.subplots()

    for i in range(4):
        package = packages[f"package_{i}"]
        phase = package[1]
        err_p = package[3]
        y_model_p = y_models_p[i]

        ax.errorbar(spacing, phase, yerr=err_p, c=colors[i], capsize=4,linestyle=None,
                    label=f"Frequencies {frequencies[i]}s")

        # Plot fitted model
        ax.plot(spacing, y_model_p, c=colors[i])

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

    if show:
        plt.show()

    plt.close(fig)