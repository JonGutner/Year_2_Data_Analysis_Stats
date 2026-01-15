import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from Radiation_Lab import r_helper, r_pdfs

def plot_histogram(energy_file, data_name, electron_data = True):
        weights = np.ones_like(energy_file) / len(energy_file)
        fig = plt.figure()

        plt.hist(energy_file, bins=100, weights=weights)
        plt.xlabel("Energy (keV)")
        if electron_data:
            plt.ylabel("Fraction of electrons")
            plt.title(f"Energy deposited by electrons of energy {data_name}")
        else:
            plt.ylabel("Fraction of photons")
            data_name_gamma = data_name.replace("_gamma", "")
            plt.title(f"Energy deposited by photons of energy {data_name_gamma}")

        return fig

def plot_exponential_fits(df, bin_widths, save=True, plot=True):
    file_name = "Histogram5.png"
    folder_name = "Histogram"

    x = df.iloc[:, 0].to_numpy(dtype=float)
    x_fit = np.linspace(min(x), max(x), 400)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    yerr = np.sqrt(y)
    if bin_widths is not None:
        xerr = bin_widths / 2
    else:
        xerr = None

    popt, pcov = curve_fit(r_pdfs.exponential_decay, x, y, p0=[max(y), 100, min(y)], sigma=yerr, absolute_sigma=True, maxfev=10000)
    a_fit, tau_fit, c_fit = popt

    fig, ax = plt.subplots()

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        xerr=xerr,
        linestyle='None',
        label='Observed'
    )
    ax.plot(x_fit, r_pdfs.exponential_decay(x_fit, *popt), 'r-',
            label=f'Exponential fit\nA={a_fit:.2f}, tau={tau_fit:.2f},c={c_fit:.2f}')
    ax.set_xlabel('Interval between Events (μs)')
    ax.set_ylabel('Number of cycles')
    ax.legend()
    fig.tight_layout()

    if save:
        r_helper.save_plot(fig, file_name, folder_name)
    if plot:
        plt.show()

    plt.close(fig)

def plot_poisson_fits(x, n_obs, bin_widths, n_exp, mu_hat, sigma_hat, chi2_red, file_name, folder_name, save=True, plot=True):
    fig, ax = plt.subplots()

    # Vertical (Poisson) errors
    yerr = np.sqrt(n_obs)

    # Horizontal bin-width indicators
    if bin_widths is not None:
        xerr = bin_widths / 2
    else:
        xerr = None

    ax.errorbar(
        x,
        n_obs,
        yerr=yerr,
        xerr=xerr,
        fmt='o',
        label='Observed'
    )

    ax.plot(
        x,
        n_exp,
        'r-',
        label=(
            rf'Poisson fit '
            rf'($\mu={mu_hat:.2f},\ \sigma={sigma_hat:.2f}$)'
            f'\nχ²/ndof = {chi2_red:.2f}'
        )
    )

    ax.set_xlabel('Counts per cycle')
    ax.set_ylabel('Number of cycles')
    ax.legend()
    fig.tight_layout()

    if save:
        r_helper.save_plot(fig, file_name, folder_name)
    if plot:
        plt.show()

    plt.close(fig)