#Main method where all the setups are performed.
import numpy as np
from Year_2_Stats import helpers, pdfs
from Waves_Lab import data_management

# User settings
data_folder = "Waves_Stat_Folder"        # <--- change this to switch datasets
# chosen_pdf = pdfs.sine_with_phase    # <--- change to exponential, poisson_pmf, etc.

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
data_name_1 = "period_1.csv"
df_0_1, df_1_1, df_2_1, df_3_1, df_4_1, df_5_1 = data_management.send_data(data_name_1)
data_name_2 = "period_2.csv"
df_0_2, df_1_2, df_2_2, df_3_2, df_4_2, df_5_2 = data_management.send_data(data_name_2)
data_name_3 = "period_3.csv"
df_0_3, df_1_3, df_2_3, df_3_3, df_4_3, df_5_3 = data_management.send_data(data_name_3)
data_name_4 = "period_4.csv"
df_0_4, df_1_4, df_2_4, df_3_4, df_4_4, df_5_4 = data_management.send_data(data_name_4)

# -----------------------------
# Run tests
# helpers.run_tests_pdf(data_frames, chosen_pdf, param_names, data_folder)

# Run tests for Waves Experiment
amplitudes_1, phases_1, err_a_1, err_p_1 = (helpers.get_ampli_phase_err_waves
                                    (df_0_1, df_1_1, df_2_1, df_3_1, df_4_1, df_5_1, pdfs.sine_with_phase_15, param_names, 0))
amplitudes_2, phases_2, err_a_2, err_p_2 = (helpers.get_ampli_phase_err_waves
                                    (df_0_2, df_1_2, df_2_2, df_3_2, df_4_2, df_5_2, pdfs.sine_with_phase_20, param_names, 1))
amplitudes_3, phases_3, err_a_3, err_p_3 = (helpers.get_ampli_phase_err_waves
                                    (df_0_3, df_1_3, df_2_3, df_3_3, df_4_3, df_5_3, pdfs.sine_with_phase_30, param_names, 2))
amplitudes_4, phases_4, err_a_4, err_p_4 = (helpers.get_ampli_phase_err_waves
                                    (df_0_4, df_1_4, df_2_4, df_3_4, df_4_4, df_5_4, pdfs.sine_with_phase_60, param_names, 3))

phases_1 = phases_1 - phases_1[0]
phases_1[4] = phases_1[4] + np.pi
phases_1[5] = phases_1[5] + np.pi
phases_2 = phases_2 - phases_2[0]
phases_3 = phases_3 - phases_3[0]
phases_4 = phases_4 - phases_4[0]

package_0 = [amplitudes_1, phases_1, err_a_1, err_p_1]
package_1 = [amplitudes_2, phases_2, err_a_2, err_p_2]
package_2 = [amplitudes_3, phases_3, err_a_3, err_p_3]
package_3 = [amplitudes_4, phases_4, err_a_4, err_p_4]
packages = {"package_0" : package_0,
            "package_1" : package_1,
            "package_2" : package_2,
            "package_3" : package_3}

helpers.run_waves_plots(packages)