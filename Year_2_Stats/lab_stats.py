#Main method where all the setups are performed.
from . import helpers, estimators, outputer

import pandas as pd
import numpy as np
import glob, os

#Reads the data in a folder, and stores it as a list of dataframes
data_folder = "Data_1" #<------Change here to read different data

current_dir = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join(current_dir, "..", data_folder)
folder_path = os.path.abspath(folder_path)

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
data_frames = [pd.read_csv(file) for file in csv_files]