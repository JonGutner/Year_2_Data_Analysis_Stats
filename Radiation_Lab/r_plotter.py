import matplotlib.pyplot as plt
import numpy as np

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