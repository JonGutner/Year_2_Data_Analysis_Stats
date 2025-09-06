#Outputs any results that come of the analysis.
import matplotlib.pyplot as plt
import numpy as np

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

def show_fit(data, pdf, params, bins=50, title="MLE Fit"):
    """Histogram + fitted PDF overlay."""
    plt.hist(data, bins=bins, density=True, alpha=0.5, label="Data")

    x = np.linspace(min(data), max(data), 200)
    y = pdf(x, *params)
    plt.plot(x, y, "r-", label="Fitted PDF")

    plt.title(title)
    plt.legend()
    plt.show()
