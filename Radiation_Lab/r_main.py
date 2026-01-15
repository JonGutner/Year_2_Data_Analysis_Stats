from Radiation_Lab import r_helper

data_names = ["2MeV.txt", "2MeV_gamma.txt", "60keV_gamma.txt", "300keV.txt"] # Add the names of the files needing analysis
poisson_names = ["Histogram1.csv", "Histogram2.csv", "Histogram3.csv", "Histogram4.csv"]

# energy_files = r_helper.data_loader(data_names, False)
# r_helper.run_histogram(energy_files, data_names, False)

poisson_files = r_helper.data_loader(poisson_names)
r_helper.run_histogram(poisson_files, poisson_names)