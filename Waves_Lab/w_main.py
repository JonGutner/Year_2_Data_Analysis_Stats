#Main method where all the setups are performed.
import numpy as np
from Waves_Lab import data_management, w_pdfs, w_helpers

# User settings
data_folder = "Waves_Stat_Folder"        # <--- change this to switch datasets
chosen_pdf = w_pdfs.sine_with_phase    # <--- change to exponential, poisson_pmf, etc.

# For Waves Experiment
param_names = ["amplitude", "phase", "c"]

# -----------------------------
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
# Run tests for Waves Experiment
amplitudes_1, phases_1, err_a_1, err_p_1 = (w_helpers.get_ampli_phase_err
                                    (df_0_1, df_1_1, df_2_1, df_3_1, df_4_1, df_5_1, w_pdfs.sine_with_phase_15, param_names, 0))
amplitudes_2, phases_2, err_a_2, err_p_2 = (w_helpers.get_ampli_phase_err
                                    (df_0_2, df_1_2, df_2_2, df_3_2, df_4_2, df_5_2, w_pdfs.sine_with_phase_20, param_names, 1))
amplitudes_3, phases_3, err_a_3, err_p_3 = (w_helpers.get_ampli_phase_err
                                    (df_0_3, df_1_3, df_2_3, df_3_3, df_4_3, df_5_3, w_pdfs.sine_with_phase_30, param_names, 2))
amplitudes_4, phases_4, err_a_4, err_p_4 = (w_helpers.get_ampli_phase_err
                                    (df_0_4, df_1_4, df_2_4, df_3_4, df_4_4, df_5_4, w_pdfs.sine_with_phase_60, param_names, 3))

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

w_helpers.run_waves_plots(packages)