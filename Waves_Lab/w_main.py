#Main method where all the setups are performed.
import numpy as np
from Waves_Lab import w_pdfs, w_helpers

# User settings
data_folder = "Waves_Stat_Folder"        # <--- change this to switch datasets
chosen_pdf = w_pdfs.sine_with_phase
# For Waves Experiment
param_names = ["amplitude", "phase", "c"]
# Periods used for the experiment
periods = np.array([15, 30, 60])

# -----------------------------
#Runs thermal waves analysis
w_helpers.load_run_thermal_vs_electrical(param_names, periods, True)

#Runs electrical waves analysis
# w_helpers.load_run_thermal_vs_electrical(param_names, periods, False)