from pathlib import Path
from Waves_Lab import load_dataset
import matplotlib.pyplot as plt

data_path = Path(__file__).resolve().parent.parent / "Waves_Lab_Data" / "Test_1.csv"

timestamp, output_voltage, output_current, thermistor_temperatures, comments = load_dataset.load(data_path)

print(thermistor_temperatures.shape)
print(timestamp.shape)
print(thermistor_temperatures[:,0].shape)

plt.plot(timestamp, thermistor_temperatures[:,0], c='r')
plt.plot(timestamp, thermistor_temperatures[:,3], c='b')
plt.show()