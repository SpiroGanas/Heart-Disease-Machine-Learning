# Spiro Ganas
# 11/2/17
#
# This project shows my level of experiance using the Python's sklearn machine learning library
#
# The data set is from:  http://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
#


import os
import requests
import pandas as pd



# Step 1: Download the data using the requests library
def download_file(url, save_folder):
    '''If the file isn't already in the save folder, download it.'''
    local_filename = url.split('/')[-1]  #Get the name of the file being downloaded

    # if the file exists, don't download it again
    if os.path.isfile(os.path.join(save_folder, local_filename)): return os.path.join(save_folder, local_filename)

    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(os.path.join(save_folder, local_filename), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return os.path.join(save_folder, local_filename)  #Returns the path pointing to the local file




def load_heart_disease_dataframe(save_data = False, print_details = False):
    """Downloads the four csv files from the UCI Machine Learning Repository website.  Loads the data

    """

    # Step 1: download the files and create a varaible pointing to the local copy of the csv file.
    cleveland = download_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", os.getcwd())
    switzerland = download_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        os.getcwd())
    va = download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
                       os.getcwd())
    hungarian = download_file(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data", os.getcwd())

    # Step 2: load the data into pandas DataFrames and then merge the four files into a single dataframe
    cleveland_df = pd.read_csv(cleveland, header=None, na_values =["?", -9.0])
    switzerland_df = pd.read_csv(switzerland, header=None, na_values =["?", -9.0])
    va_df = pd.read_csv(va, header=None, na_values =["?", -9.0])
    hungarian_df = pd.read_csv(hungarian, header=None, na_values =["?", -9.0])

    #add a column to keep track of the source of the data
    cleveland_df["Source"] = "cleveland"
    switzerland_df["Source"] = "switzerland"
    va_df["Source"] = "va"
    hungarian_df["Source"] = "hungarian"


    # add headers to the data frames
    headers = {0 : "age",
               1 : "sex",
               2 : "cp",
               3 : "trestbps",
               4 : "chol",
               5 : "fbs",
               6 : "restecg",
               7 : "thalach",
               8 : "exang",
               9 : "oldpeak",
               10 : "slope",
               11 : "ca",
               12 : "thal",
               13 : "diagnosis"}

    cleveland_df = cleveland_df.rename(columns=headers)
    switzerland_df = switzerland_df.rename(columns=headers)
    va_df = va_df.rename(columns=headers)
    hungarian_df = hungarian_df.rename(columns=headers)

    # append the four files into a single dataframe
    heart_disease_df = cleveland_df.append(switzerland_df).append(va_df).append(hungarian_df)




    #print some basic info about the data
    if print_details:
        print(heart_disease_df.head())
        print()
        print(heart_disease_df.info())
        print()


    # Save the merged data as a csv
    if save_data:  heart_disease_df.to_csv(os.path.join(os.getcwd(), "heart_disease_df.csv"))


if __name__ == '__main__':
    heart_disease_df = load_heart_disease_dataframe(save_data = True, print_details = True)
    print('The heart disease data has been downloaded!')










