# Year 2 Stats and Data Analysis

## Dev requirements & dependencies

* Python version: 3.13
* See requirements.txt for more information

## Quick Note on How to Switch to Analyse the Correct Thing
* In main.py (for pdfs) or w_main.py (for waves experiment), make sure to have the correct parts (un)commented, or defined.
* For example in the main.py, make sure to have the 'gaussian' pdf chosen if the analysis is of a gaussian distribution.
* Make sure 'data_folder' variable is the data folder's name that contains the data you want to analyse.
* Make sure to comment any parameter names of the pdfs you don't use, and vice versa, uncomment the parameter names of the pdf you use.

## Year 2 Lab
### Waves Lab (use of PDFs after this)

What it does (Thermal Waves)
* Reads .csv files containing the data.
* Graphs thermistors 0-7 together on one plot.
* Uses MLE & analyses goodness of fit (in regression) of the steady state.
* Prints out the fitted parameter's values, uncertainties, goodness of fit and chi-squared information.
* Saves the fitted graphs on the desktop (saving of the plot of all thermistors plot hasn't been added yet).
* Produces plots for how the phase and amplitude changes for each thermistor. Includes fits for the old and new models.

Usage
1. Modify w_main.py to run thermal waves by uncommenting line 15, and commenting line 18.
2. Add data to a data folder. Make sure to modify w_main.py's data_folder name to the folder with the data. 
3. For fitting of data, make sure to modify periods in w_main.py to the periods used, AND make sure w_pdfs.py contains the function for that period.
4. To run the code, press the green run button in w_main.py. DON'T use the console, it won't work due to properties of relative imports.

What it does (Electrical Waves)
* NOTE: Thermal waves part is better written for generalisation and correctness, due to lack of focus on this part. 
* Reads .csv files containing the data.
* Does the code for the tasks related to electrical waves (apologies for low detail explanation) including plots and value fitting.

Usage
1. Modify w_main.py to run electrical waves by uncommenting line 18, and commenting line 15.
2. Similar to thermal waves, make sure you use the appropriate variable definition for the data_folder in w_main.py.
3. Make sure w_pdfs.py contains the function with the used frequency.
4. THIS SECTION MAY REQUIRE CODE MODIFICATIONS DUE TO LACK OF GENERALISATION.
5. Run the code by pressing the green run button.

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
   
### What it does

* Uses MLE and goodness of fit to make the analysis of those PDFs.
* Produces graphs that are saved onto the desktop.

### Unittests

* There is a class called tests.py in the tests package.
* tests.py tests whether the code is running properly for PDF analysis.
* To test if it is running properly, press the green run button (control F5) while having open the 'tests.py' class open.
