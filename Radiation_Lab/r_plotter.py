import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

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

def plot_exponential(df, data_name, save=False, plot=True):
    folder_name = "Exponential"

    x = df.iloc[:, 1].to_numpy(dtype=float)
    y = df.iloc[:, 2].to_numpy(dtype=float)
    yerr = df.iloc[:, 4].to_numpy(dtype=float)
    xerr = 0.01

    popt, pcov = curve_fit(
        r_pdfs.exponential_decay,
        x, y,
        p0=[max(y), 100, min(y)],
        sigma=yerr,
        absolute_sigma=True,
        maxfev=10000
    )

    a_fit, tau_fit, c_fit = popt
    y_fit = r_pdfs.exponential_decay(x, *popt)

    # Normalised residuals
    residuals = (y - y_fit) / yerr

    # --- Figure with residuals ---
    fig, (ax, ax_res) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Main plot
    ax.errorbar(
        x, y,
        yerr=yerr,
        xerr=xerr,
        linestyle='None',
        label='Observed'
    )

    ax.plot(
        x, y_fit,
        label=f'Exponential fit\nA={a_fit:.2f}, τ={tau_fit:.2f}, c={c_fit:.2f}'
    )

    ax.set_ylabel('Number of cycles')
    ax.set_yscale('log')
    ax.legend()

    # Residual plot
    ax_res.errorbar(
        x, residuals,
        yerr=np.ones_like(residuals),
        xerr=xerr,
        linestyle='None'
    )

    ax_res.axhline(0)
    ax_res.set_xlabel('Copper Thickness (mm)')
    ax_res.set_ylabel('Residuals\n$(y - y_{fit})/\\sigma$')

    fig.tight_layout()

    if save:
        r_helper.save_plot(fig, data_name, folder_name)
    if plot:
        plt.show()

    plt.close(fig)

def plot_exponential_binned(df, bin_widths, save=True, plot=True):
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

def plot_nd_dt(df, folder_name, file_name, save=False, plot=True):
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    yerr = df.iloc[:, 2].to_numpy(dtype=float)
    xerr = 0.001

    popt, pcov = curve_fit(r_pdfs.linear_flat, x, y, p0=[0, (max(y) + min(y)) / 2], sigma=yerr, absolute_sigma=True,
                           maxfev=10000)
    m_fit, c_fit = popt

    y_fit = r_pdfs.linear_flat(x, *popt)

    # Chi-squared
    chi2 = np.sum(((y - y_fit) / yerr) ** 2)

    # Degrees of freedom
    ndof = len(y) - len(popt)

    # Reduced chi-squared
    chi2_red = chi2 / ndof

    print(f"Chi-squared = {chi2:.2f}")
    print(f"Reduced chi-squared = {chi2_red:.2f}")
    print(f"Degrees of freedom = {ndof}")

    fig, ax = plt.subplots()

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        xerr=xerr,
        fmt='o',
        label='Data'
    )

    ax.plot(
        x,
        y_fit,
        'r-',
        label=(
            f'Linear fit\n'
            f'm = {m_fit:.2f}, c = {c_fit:.2f}\n'
            f'χ²ᵣ = {chi2_red:.2f}'
        )
    )

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('nd^2/Δt')
    ax.set_title("Task 16 plot")
    ax.legend()

    if save:
        r_helper.save_plot(fig, file_name, folder_name)
    if plot:
        plt.show()

    plt.close(fig)

