#Main method where all the setups are performed.
from Year_2_Stats import helpers, pdfs
from Waves_Lab import data_management

# User settings
data_folder = "Waves_Stat_Folder"        # <--- change this to switch datasets
chosen_pdf = pdfs.sine_with_phase    # <--- change to exponential, poisson_pmf, etc.

# Parameter names for all PDFs (comment/uncomment as needed)
# For Gaussian
# param_names = ["mu", "sigma"]

# For Exponential
# param_names = ["lambda"]

# For Poisson
# param_names = ["mu"]

# For Binomial (note: n must usually be fixed, not fitted)
# param_names = ["p"]

# For Lorentzian
# param_names = ["x0", "gamma"]

# For Uniform
# param_names = ["a", "b"]

# For Waves Experiment
param_names = ["amplitude", "phase", "c"]

# -----------------------------
# Load data
# data_frames = helpers.load_data(data_folder)

# Load data for Waves Experiment
df_0, df_1, df_2, df_3 = data_management.send_data()
# print(df_0)

# -----------------------------
# Run tests
# helpers.run_tests_pdf(data_frames, chosen_pdf, param_names, data_folder)

# Run tests for Waves Experiment
amplitudes = []
phases = []
err_a = []
err_p = []

results, errs = helpers.run_tests_waves(df_0, 0, chosen_pdf, param_names)
amplitudes.append(results[0])
phases.append(results[1])
err_a.append(errs[0])
err_p.append(errs[1])
results, errs = helpers.run_tests_waves(df_1, 1, chosen_pdf, param_names)
amplitudes.append(results[0])
phases.append(results[1])
err_a.append(errs[0])
err_p.append(errs[1])
results, errs = helpers.run_tests_waves(df_2, 2, chosen_pdf, param_names)
amplitudes.append(results[0])
phases.append(results[1])
err_a.append(errs[0])
err_p.append(errs[1])
results, errs = helpers.run_tests_waves(df_3, 3, chosen_pdf, param_names)
amplitudes.append(results[0])
phases.append(results[1])
err_a.append(errs[0])
err_p.append(errs[1])

helpers.run_waves_plots(amplitudes, phases, err_a, err_p)