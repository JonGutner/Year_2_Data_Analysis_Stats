import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Radiation_Lab import r_plotter

def data_loader(data_names):
    energy_files = []
    for data_name in data_names:
        data_path = Path(__file__).resolve().parent.parent / "radiation_data_folder" / data_name

        if not data_path.exists():
            raise RuntimeError(f"Directory does not exist: {data_path}")

        txt_file = pd.read_csv(data_path, delimiter=",", skiprows=1, usecols=[0])

        if len(txt_file) == 0:
            raise RuntimeError("No CSV files found in the directory.")

        energy_files.append(txt_file)

    return energy_files

def remove_txt(data_name):
    new_data_name = data_name.replace(".txt", "")
    return new_data_name

def run_histogram(energy_files, data_names):
    for i in range(len(energy_files)):
        data_name = remove_txt(data_names[i])
        if "gamma" in data_name:
            fig = r_plotter.plot_histogram(energy_files[i], data_name, False)
            save_plot(fig, f"{data_name}.png", "Task_8_Plots")
        else:
            fig = r_plotter.plot_histogram(energy_files[i], data_name, True)
            save_plot(fig, f"{data_name}.png", "Task_7_Plots")

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