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

def send_data_thermal(data_name):
    data_path = Path(__file__).resolve().parent.parent / "Waves_Stat_Folder" / data_name

    timestamp, output_voltage, output_current, thermistor_temperatures, comments = load_data(data_path)

    plt.title("Plot of Thermistors 0-7")
    plt.plot(timestamp, thermistor_temperatures[:, 0], c='red', label="Therm. 0")
    plt.plot(timestamp, thermistor_temperatures[:, 1], c='blue', label="Therm. 1")
    plt.plot(timestamp, thermistor_temperatures[:, 2], c='green', label="Therm. 2")
    plt.plot(timestamp, thermistor_temperatures[:, 3], c='orange', label="Therm. 3")
    plt.plot(timestamp, thermistor_temperatures[:, 4], c='cyan', label="Therm. 4")
    plt.plot(timestamp, thermistor_temperatures[:, 5], c='purple', label="Therm. 5")
    plt.plot(timestamp, thermistor_temperatures[:, 6], c='gold', label="Therm. 6")
    plt.plot(timestamp, thermistor_temperatures[:, 7], c='lime', label="Therm. 7")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
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
    df_4 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 4]})
    df_5 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 5]})
    df_6 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 6]})
    df_7 = pd.DataFrame({
        't': timestamp,
        'T': thermistor_temperatures[:, 7]})

    return df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7

def send_data_electrical(data_name):
    data_path = Path(__file__).resolve().parent.parent / "Electrical_Waves_Sine_Data" / data_name

    if not data_path.exists():
        raise RuntimeError(f"Directory does not exist: {data_path}")

    csv_files = [f for f in data_path.iterdir() if f.suffix.lower() == ".csv"]

    if len(csv_files) == 0:
        raise RuntimeError("No CSV files found in the directory.")

    # Extract group ID and channel number
    pattern = re.compile(r".*?_(\d+)_([123])\.csv$", re.IGNORECASE)

    groups = {}
    for f in csv_files:
        match = pattern.match(f.name)
        if not match:
            print(f"Skipping unrecognized file name format: {f.name}")
            continue

        group_id, ch = match.groups()
        ch = int(ch)

        if group_id not in groups:
            groups[group_id] = {}

        groups[group_id][ch] = pd.read_csv(f)

    combined_outputs = {}

    for group_id, channels in groups.items():

        # Skip incomplete groups
        if set(channels.keys()) != {1, 2, 3}:
            print(f"Skipping incomplete group {group_id}, missing channels.")
            continue

        # Normalize headers
        for ch in channels.values():
            ch.columns = [c.strip().lower() for c in ch.columns]

        # --- NEW: Time alignment check ---
        t1 = channels[1]["in s"].iloc[0]
        t2 = channels[2]["in s"].iloc[0]
        t3 = channels[3]["in s"].iloc[0]

        if not (t1 == t2 == t3):
            print(f"**WARNING**: Time start mismatch in group {group_id}: "
                  f"Ch1={t1}, Ch2={t2}, Ch3={t3}")

        # Build combined DataFrame
        combined = pd.DataFrame({
            "time": channels[1]["in s"],
            "ch1": channels[1]["c1 in v"],
            "ch2": channels[2]["c2 in v"],
            "ch3": channels[3]["c3 in v"],
        })

        combined_outputs[group_id] = combined

    return combined_outputs