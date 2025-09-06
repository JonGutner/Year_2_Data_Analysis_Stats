#Main method where all the setups are performed.
from . import helpers, pdfs

# User settings
data_folder = "Data_1"        # <--- change this to switch datasets
chosen_pdf = pdfs.gaussian    # <--- change to exponential, poisson_pmf, etc.

# Parameter names for all PDFs (comment/uncomment as needed)
# For Gaussian
param_names = ["mu", "sigma"]

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

# -----------------------------
# Load data
data_frames = helpers.load_data(data_folder)

# -----------------------------
# Run tests
helpers.run_tests(data_frames, chosen_pdf, param_names)
