import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(energy_files, data_names):
    for i in range(len(energy_files)):
        plt.hist(energy_files[i], bins=100)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Count")
        plt.title(data_names[i])
        plt.show()