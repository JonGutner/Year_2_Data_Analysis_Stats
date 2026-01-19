from Radiation_Lab import r_helper

data_names = ["2MeV.txt", "2MeV_gamma.txt", "60keV_gamma.txt", "300keV.txt"] # Add the names of the files needing analysis
poisson_names = ["Histogram1.csv", "Histogram2.csv", "Histogram5.csv"]
nd_dt = "Task_16.csv"

# energy_files = r_helper.data_loader(data_names, False)
# r_helper.run_histogram(energy_files, data_names, False)

# poisson_files = r_helper.data_loader(poisson_names)
# r_helper.run_histogram(poisson_files, poisson_names)

nd_dt_files = []
nd_dt_files.append(r_helper.nd_dt_data_loader(nd_dt, 1))
nd_dt_files.append(r_helper.nd_dt_data_loader(nd_dt, 7))
r_helper.run_nd_dt(nd_dt_files, ["All Data of Task 16", "Linear Part of Task 16"])