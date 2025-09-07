#Outputs any results that come of the analysis.
import matplotlib.pyplot as plt
import numpy as np
import os

def print_results(name, result, param_names=None):
    print(f"\nResults for {name}:")
    for i, val in enumerate(result["params"]):
        name_str = param_names[i] if param_names else f"param{i}"
        fisher_err = result["fisher_errors"][i]
        low, high = result["profile_intervals"][i]
        print(f"  {name_str}: {val:.4f} ± {fisher_err:.4f} "
              f"(68% CL: {low:.4f} – {high:.4f})")
    print("  -logL:", result["neg_logL"])
    print("  Converged:", result["success"])

def show_fit(data, pdf, params, bins=50, title="MLE Fit", save=True, show=False):
    """Histogram + fitted PDF overlay. Optionally save to Desktop/StatsPlots."""
    fig, ax = plt.subplots()

    # Histogram of the data
    ax.hist(data, bins=bins, density=True, alpha=0.5, label="Data")

    # Fitted PDF
    x = np.linspace(min(data), max(data), 200)
    y = pdf(x, *params)
    ax.plot(x, y, "r-", label="Fitted PDF")

    ax.set_title(title)
    ax.legend()

    # Save
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
