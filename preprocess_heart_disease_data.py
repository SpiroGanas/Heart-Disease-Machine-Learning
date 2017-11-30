# Spiro Ganas
# 11/2/17
#
# This project shows my level of experience using the Python's sklearn machine learning library
#
# The data set is from:  http://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
#


# Notes about the data
# age          continuous: age in years
# ca           discrete: number of major vessels (0-3) colored by flourosopy
# chol         continuous: serum cholesterol in mg/dl
# cp           discrete: chest pain type
#                        -- Value 1: typical angina
#                        -- Value 2: atypical angina
#                        -- Value 3: non-anginal pain
#                        -- Value 4: asymptomatic
# diagnosis    discrete: diagnosis of heart disease (angiographic disease status)
#                        -- Value 0: < 50% diameter narrowing
#                        -- Value 1: > 50% diameter narrowing
# exang        discrete: exercise induced angina (1 = yes; 0 = no)
# fbs          discrete: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
# oldpeak      continuous: ST depression induced by exercise relative to rest
# restecg      discrete: resting electrocardiographic results
#                        -- Value 0: normal
#                        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#                        -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# sex          discrete: sex (1 = male; 0 = female)
# slope        discrete: the slope of the peak exercise ST segment
#                        -- Value 1: upsloping
#                        -- Value 2: flat
#                        -- Value 3: downsloping
# thal         discrete: 3 = normal; 6 = fixed defect; 7 = reversable defect
# thalach      continuous: maximum heart rate achieved
# trestbps     continuous:resting blood pressure (in mm Hg on admission to the hospital)








import os
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import numpy as np

# turn off warning messages that were messing up the output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





def Preprocess_Heart_Disease_Data():
    CONTINUOUS_FACTORS = ["age", "chol", "oldpeak", "thalach", "trestbps"]
    DISCRETE_FACTORS = ["ca", "cp", "exang", "fbs", "restecg", "sex", "slope", "thal"]
    TARGET = ["diagnosis"]

    # load the csv file created by load_heart_disease_df.py
    heart_disease_df = pd.read_csv(os.path.join(os.getcwd(),"heart_disease_df.csv" ),index_col=0, header=0 )





    # A quick look at the data
    #print(heart_disease_df.info())
    #print()
    # Descriptive statistics on the data
    #print(heart_disease_df.describe())

    # use sklearn to split the data into training and testing sets
    train, test = sklearn.model_selection.train_test_split(heart_disease_df, test_size=.2, random_state=10)


    # split the data into the X and Y variables
    trainY = train["diagnosis"].copy()
    trainX = train.drop("diagnosis", axis=1)
    testY = test["diagnosis"].copy()
    testX = test.drop("diagnosis", axis=1)


    # Handle the discrete and continuous variables seperatly
    trainX_continuous = trainX[CONTINUOUS_FACTORS]
    trainX_discrete = trainX[DISCRETE_FACTORS].fillna(value=5) #For all the discrete values, replace missing values with 5
    testX_continuous = testX[CONTINUOUS_FACTORS]
    testX_discrete = testX[DISCRETE_FACTORS].fillna(value=5) #For all the discrete values, replace missing values with 5


    # For continuous variables, replace missing values with the median and then normalize by subtracting the mean and dividing by the standard deviation
    continuous_Pipeline = sklearn.pipeline.Pipeline( [("imputer", sklearn.preprocessing.Imputer(strategy="median")),
                                             ("scaler", sklearn.preprocessing.StandardScaler())
                                            ]
                                           )
    trainX_continuous_scaled = continuous_Pipeline.fit_transform(trainX_continuous)

    #print(trainX_discrete.info())

    #print(trainX_discrete)
    #enc = sklearn.preprocessing.OneHotEncoder(handle_unknown ='ignore')
    #trainX_discrete_one_hot = enc.fit_transform(trainX_discrete)

    # for discrete variables, one-hot encode the data
    discrete_Pipeline = sklearn.pipeline.Pipeline( [("one_hot", sklearn.preprocessing.OneHotEncoder(handle_unknown ='ignore'))
                                            ]
                                           )
    trainX_discrete_one_hot = discrete_Pipeline.fit_transform(trainX_discrete)



    ## This will only work if I use the dataframe selector code on page 67
    #merged_pipeline = sklearn.pipeline.FeatureUnion(transformer_list=[('continuous_Pipeline',continuous_Pipeline),
    #                                                                  ('discrete_Pipeline', discrete_Pipeline)
    #                                                                    ])
    #merged_pipeline.fit()


    trainX_fully_preprocessed = np.concatenate((trainX_continuous_scaled, trainX_discrete_one_hot.toarray()), axis = 1)

    #### THIS is the final X dataset that will be used to build the predictive model
    #print(trainX_fully_preprocessed)




    # preprocess the test data, note the use of transform instead of fit_transform
    testX_continuous_scaled = continuous_Pipeline.transform(testX_continuous)
    testX_discrete_one_hot = discrete_Pipeline.transform(testX_discrete)
    testX_fully_preprocessed = np.concatenate((testX_continuous_scaled, testX_discrete_one_hot.toarray()), axis = 1)




    # Convert the Y varialbe to a binary varible
    trainY_binary = (trainY>0) # True if they have heart disease, False otherwide
    testY_binary = (testY>0) # True if they have heart disease, False otherwide

    return trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary




