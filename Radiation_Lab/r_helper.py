import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize

from Radiation_Lab import r_plotter
from Year_2_Stats import helpers, pdfs

def data_loader(data_names, all_data=True):
    energy_files = []
    for data_name in data_names:
        data_path = Path(__file__).resolve().parent.parent / "radiation_data_folder" / data_name

        if not data_path.exists():
            raise RuntimeError(f"Directory does not exist: {data_path}")

        if all_data:
            txt_file = pd.read_csv(data_path, delimiter=",", skiprows=1)
        else:
            txt_file = pd.read_csv(data_path, delimiter=",", skiprows=1, usecols=[0])

        if len(txt_file) == 0:
            raise RuntimeError("No CSV files found in the directory.")

        energy_files.append(txt_file)

    return energy_files

def remove_txt(data_name):
    new_data_name = data_name.replace(".txt", "")
    return new_data_name

def poisson_fit_histogram(
    df, file_name="poisson_fit.png",
    bin_widths=None,
    plot=True,
    save=False,
    folder_name="Poisson_Fits"
):
    # -----------------------------
    # Extract data
    # -----------------------------
    x = df.iloc[:, 0].to_numpy(dtype=float)      # bin centres
    n_obs = df.iloc[:, 1].to_numpy(dtype=float)  # counts per bin
    Ntot = np.sum(n_obs)

    # -----------------------------
    # Poisson negative log-likelihood
    # -----------------------------
    def nll(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf

        pdf = norm.pdf(x, mu, sigma)
        mu_i = Ntot * pdf / np.sum(pdf)

        return np.sum(mu_i - n_obs * np.log(mu_i + 1e-12))

    # Initial guesses
    mu0 = np.average(x, weights=n_obs)
    sigma0 = np.sqrt(mu0)

    result = minimize(nll, x0=[mu0, sigma0])
    mu_hat, sigma_hat = result.x

    # -----------------------------
    # Expected counts per bin
    # -----------------------------
    pdf = norm.pdf(x, mu_hat, sigma_hat)
    n_exp = Ntot * pdf / np.sum(pdf)

    # -----------------------------
    # χ² GOES HERE (this is the correct location)
    # -----------------------------
    chi2 = np.sum((n_obs - n_exp)**2 / n_exp)
    ndof = len(n_obs) - 2
    chi2_red = chi2 / ndof

    # -----------------------------
    # Plot
    # -----------------------------
    if plot or save:
        r_plotter.plot_poisson_fits(x, n_obs, bin_widths, n_exp, mu_hat, sigma_hat, chi2_red, file_name, folder_name, save=save)

    return mu_hat, sigma_hat, chi2, ndof, chi2_red

def run_histogram(energy_files, data_names, fit_poisson = True):
    folder_name = "Radiation_Lab_Plots"

    if fit_poisson:
        fit_results = []

        for i, df in enumerate(energy_files):
            if data_names[i] != "Histogram5.csv":
                x = df.iloc[:, 0].to_numpy(dtype=float)
                bin_widths = compute_bin_widths(x)
                mu_hat, sigma_hat, chi2, ndof, chi2_red = poisson_fit_histogram(
                    df, file_name=f"{data_names[i]}.png",
                    bin_widths=bin_widths,
                    plot=True,
                    save=True,
                    folder_name=f"Histogram")

                fit_results.append({
                    "dataset": df,
                    "name": data_names[i],
                    "mu_hat": mu_hat,
                    "sigma_hat" : sigma_hat,
                    "chi2" : chi2,
                    "ndof" : ndof,
                    "chi2_red" : chi2_red
                })

                print("------------------")
                print(data_names[i])
                print("Chi2:",chi2, "DoF:", ndof, "chi2_red:", chi2_red)
                print("mu:",mu_hat, "sigma:",sigma_hat)
                print("------------------")

            else:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                bin_widths = compute_bin_widths(x)

                r_plotter.plot_exponential_fits(df, bin_widths)

    else:
        for i in range(len(energy_files)):
            data_name = remove_txt(data_names[i])

            if "gamma" in data_name:
                fig = r_plotter.plot_histogram(energy_files[i], data_name, False)
                save_plot(fig, f"{data_name}.png", folder_name)
            else:
                fig = r_plotter.plot_histogram(energy_files[i], data_name, True)
                save_plot(fig, f"{data_name}.png", folder_name)

def compute_bin_widths(x):
    """
    Compute bin widths from bin centres.
    Works for arbitrary, non-uniform binning.
    """
    x = np.asarray(x)

    edges = np.zeros(len(x) + 1)
    edges[1:-1] = 0.5 * (x[1:] + x[:-1])
    edges[0] = x[0] - (edges[1] - x[0])
    edges[-1] = x[-1] + (x[-1] - edges[-2])

    return np.diff(edges)

def save_plot(fig, file_name, folder_name, dpi=300):
    desktop_dir = Path.home() / "Desktop"
    output_dir = desktop_dir / folder_name

    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / file_name

    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)