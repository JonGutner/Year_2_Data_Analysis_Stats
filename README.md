# Year 2 Stats and Data Analysis

## Dev requirements & dependencies

* Python version: 3.13
* See requirements.txt for more information

## Quick Note on How to Switch to Analyse the Correct Thing
* In main.py, make sure to have the correct parts (un)commented, or defined.
* For example in the waves lab, make sure to have the 'sine_with_phase' pdf chosen.
* Make sure 'data_folder' variable is the data folder's name that contains the data you want to analyse.
* Make sure to comment any parameter names of the pdfs you don't use, and vice versa, uncomment the parameter names of the pdf you use.
* For statistical pdfs, make sure you use helpers.load_data(), but for waves, make sure you use data_management.send_data().
* For statistical pdfs, make sure you use helpers.run_tests_pdf, but for waves, make sure you use helpers.run_tests_waves.

## Year 2 Lab
### Waves Lab (use of PDFs after this)

What it does
* Reads .csv files containing the data.
* Graphs (currently for the 0-3) thermistors together on one plot.
* Uses MLE & analyses goodness of fit (in regression) of the steady state.
* Prints out the fitted parameter's values, uncertainties, goodness of fit and chi-squared information.
* Saves the fitted graphs on the desktop (saving of the plot of all thermistors plot hasn't been added yet).

Usage
1. Load the .csv file you wish to analyse in 'Waves_Stat_Folder' folder (or any folder name if you change 'data_folder' variable to hold a different string in main.py
   (one .csv file at a time in that folder for waves). You can keep the rest of the .csv files in 'General_Waves_Data' or any other folder you so choose.
3. Run by pressing green run button (control F5) while having open the 'main.py' class (found inside the 'Year_2_Stats' package).

## Statistics
### PDFs

The stats toolbox supports analysis of the following PDFs
* Gaussian
* Exponential
* Poisson
* Binomial
* Lorentzian

Usage 
1. Change 'data_folder' variable to the name of the folder containing the data (can contain multiple .csv files at a time, and while cycle through them while doing analysis).
2. Run by pressing green run button (control F5) while having open the 'main.py' class (found inside the 'Year_2_Stats' package).
   
