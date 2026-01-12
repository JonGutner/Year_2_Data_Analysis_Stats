import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(energy_file, data_name, electron_data = True):
        weights = np.ones_like(energy_file) / len(energy_file)

        plt.hist(energy_file, bins=100, weights=weights)
        plt.xlabel("Energy (keV)")
        if electron_data:
            plt.ylabel("Fraction of electrons")
        else:
            plt.ylabel("Fraction of photons")
        plt.title(data_name)
        plt.show()