import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

def load_data(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()

    comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]

    return timestamp, output_voltage, output_current, thermistor_temperatures, comments

def send_data():
    data_path = Path(__file__).resolve().parent.parent / "Waves_Stat_Folder" / "steady_state_2.csv"

    timestamp, output_voltage, output_current, thermistor_temperatures, comments = load_data(data_path)

    plt.title("Plot of Thermistors 0-3")
    plt.plot(timestamp, thermistor_temperatures[:, 0], c='r', label="Therm. 0")
    plt.plot(timestamp, thermistor_temperatures[:, 1], c='green', label="Therm. 1")
    plt.plot(timestamp, thermistor_temperatures[:, 2], c='orange', label="Therm. 2")
    plt.plot(timestamp, thermistor_temperatures[:, 3], c='b', label="Therm. 3")
    plt.legend()
    plt.show()

    df_0 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 0]})
    df_1 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 1]})
    df_2 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 2]})
    df_3 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 3]})

    return df_0, df_1, df_2, df_3