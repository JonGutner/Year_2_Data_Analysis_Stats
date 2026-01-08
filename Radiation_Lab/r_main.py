from Radiation_Lab import r_helper, r_plotter

data_names = ["2MeV.txt", "2MeV_gamma.txt", "60keV_gamma.txt", "300keV.txt"] # Add the names of the files needing analysis

energy_files = r_helper.data_loader(data_names)
r_plotter.plot_histogram(energy_files, data_names)

