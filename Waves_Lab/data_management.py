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
    data_path = Path(__file__).resolve().parent.parent / "Waves_Stat_Folder" / "Final_test.csv"

    timestamp, output_voltage, output_current, thermistor_temperatures, comments = load_data(data_path)

    print(timestamp)
    print(thermistor_temperatures[:, 0])

    plt.plot(timestamp, thermistor_temperatures[:, 0], c='r')
    plt.plot(timestamp, thermistor_temperatures[:, 3], c='b')
    plt.show()

    df_0 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 0]})
    df_3 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 3]})

    return df_0, df_3