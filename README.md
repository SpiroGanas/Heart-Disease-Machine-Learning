# Heart-Disease-Machine-Learning
sklearn machine learning models applied to the UCI Heart Disease data set.
  
The heart disease data set is from: https://archive.ics.uci.edu/ml/datasets/heart+Disease
  
[load_heart_disease_data.py](https://github.com/SpiroGanas/Heart-Disease-Machine-Learning/blob/master/load_heart_disease_data.py) is a script that downloads four csv files from the UCI website into the local working directory.  It merges the data into a single pandas dataframe, adds column headers, and (optionally) saves the data to a single csv file.
  
[preprocess_heart_disease_data.py](https://github.com/SpiroGanas/Heart-Disease-Machine-Learning/blob/master/preprocess_heart_disease_data.py) pre-processes the data by:
* Replacing missing continous variables with the media for that variable.
* Rescaling continous variables so they have a mean of 0 and a standard deviation of 1.
* One-hot encoding discrete variables (ignoring any missing values).
* Converting the heart disease column to a boolean value (True if the patient has heart disease, False otherwise).
* Splitting the data into training and testing data sets.
  
[analyze_heasrt_disease_data.py](https://github.com/SpiroGanas/Heart-Disease-Machine-Learning/blob/master/analyze_heart_disease_data.py) applies some machine learning algorithms from sklearn to the preprocessed data. Randomized Search Cross Validation is used to identify the best neural network architecture and hyperparameters.  The best "optimized" neural network model produced an F1 score of 0.857.  This compares to an F1 score of 0.79 for a neural network containing a single hidden layer with 10 nodes.
   
