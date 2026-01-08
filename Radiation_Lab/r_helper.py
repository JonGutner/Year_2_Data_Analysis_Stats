import pandas as pd
import numpy as np
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
