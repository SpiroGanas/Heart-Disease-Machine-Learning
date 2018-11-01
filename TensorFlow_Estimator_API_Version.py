# -*- coding: utf-8 -*-

#

#

# Spiro Ganas

# 2018-09-13

#

# Heart disease data set from

# https://archive.ics.uci.edu/ml/datasets/Heart+Disease

 

 

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

 

 

 

 

 

 

cleve = pd.read_csv('./processed.cleveland.data',

                    names=['age', 'sex', 'cp', 'trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'],

#                    dtype={'age':np.float16,

#                           'sex':str,

#                           'cp':str,

#                           'trestbps':np.float16,

#                           'chol':np.float16,

#                           'fbs':str,

#                           'restecg':str,

#                           'thalach':np.float16,

#                           'exang':str,

#                           'oldpeak':np.float16,

#                           'slope':str,

#                           'ca':np.float16,

#                           'thal':str,

#                           'num':str

#                           }

                    )

 

cleve.cp = cleve.cp.astype(str)

cleve.exang = cleve.exang.astype(str)

cleve.fbs = cleve.fbs.astype(str)

cleve.restecg = cleve.restecg.astype(str)

cleve.sex = cleve.sex.astype(str)

cleve.slope = cleve.slope.astype(str)

#cleve.ca = cleve.ca.astype(float)   # This column seems to contain some dirty data

 

                         

#print(cleve.head())

 

#

 

 

 

 

train, test = train_test_split(cleve, random_state =37, test_size=0.2)

 

#print(train.head())

 

 

print (test.shape)

 

 

 

 

 

#

#

#

#7. Attribute Information:

#   -- Only 14 used

#      -- 1. #3  (age)       age in years

#      -- 2. #4  (sex)       sex (1 = male; 0 = female)

#      -- 3. #9  (cp)        chest pain type

#                            -- Value 1: typical angina

#                            -- Value 2: atypical angina

#                            -- Value 3: non-anginal pain

#                            -- Value 4: asymptomatic

#      -- 4. #10 (trestbps)  resting blood pressure (in mm Hg on admission to the hospital)

#      -- 5. #12 (chol)      serum cholestoral in mg/dl

#      -- 6. #16 (fbs)       (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)

#      -- 7. #19 (restecg)   resting electrocardiographic results

#                            -- Value 0: normal

#                            -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST

#                                        elevation or depression of > 0.05 mV)

#                            -- Value 2: showing probable or definite left ventricular hypertrophy

#                                        by Estes' criteria

#      -- 8. #32 (thalach)   maximum heart rate achieved

#      -- 9. #38 (exang)     exercise induced angina (1 = yes; 0 = no)

#      -- 10. #40 (oldpeak)   ST depression induced by exercise relative to rest

#      -- 11. #41 (slope)     e 1: upsloping

#                            -- Value 2: flat

#                            -- Value 3: downsloping

#      -- 12. #44 (ca)        number of major vessels (0-3) colored by flourosopy

#      -- 13. #51 (thal)      3 = normal; 6 = fixed defect; 7 = reversable defect

#      -- 14. #58 (num)       (the predicted attribute)

#                            diagnosis of heart disease (angiographic disease status)

#                                    -- Value 0: < 50% diameter narrowing

#                                    -- Value 1: > 50% diameter narrowing

#                                    (in any major vessel: attributes 59 through 68 are vessels)

#                           

 

 

import tensorflow as tf

 

 

 

 

def createNumericFeatureColumnsList(ListOfColumnNames):

    """Takes in a list of column names as returns them as a list of numeric feature columns"""

    return [tf.feature_column.numeric_column(colName) for colName in ListOfColumnNames]

 

def createIndicatorFeatureColumnsList(ListOfColumnNamesAndVocabularies):

    """Takes in a list of tuples containing column names as a string  and vocabularies

       as a list of strings. Returns a list of indicator feature columns"""

    return [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(colName, vocabulary)) for colName, vocabulary in ListOfColumnNamesAndVocabularies]

 

 

 

 

 

 

 

 

numeric_featcols = createNumericFeatureColumnsList(["age", "trestbps", "chol", "thalach","oldpeak"])

 

# This column had some missing values

#            tf.feature_column.numeric_column("ca")

 

 

 

indicator_featcols = createIndicatorFeatureColumnsList( [("sex", ["0","1"]),

                                                         ("cp", ["1","2","3","4"]),

                                                         ("fbs", ["0","1"]),

                                                         ("restecg", ["0","1","2"]),

                                                         ("exang", ["0","1"]),

                                                         ("slope", ["1","2","3"]),

                                                         ("thal", ["3","6","7"])

                                                         ]

                                                        )

 

 

 

 

 

 

def pandas_train_input_fn(df):

    return tf.estimator.inputs.pandas_input_fn(

            x=df,

            y=df['num'],

            batch_size=128,

            num_epochs=None,

            shuffle=True,

            queue_capacity=None

            )

 

 

 

 

 

 

 

 

 

run_config = tf.estimator.RunConfig(

                                    model_dir='C:\\Spiros_Local_Files\\TF_Checkpoints\\',

                                    save_summary_steps = 5,

                                    save_checkpoints_steps = 5000

                                    )

 

 

train_spec = tf.estimator.TrainSpec(input_fn=pandas_train_input_fn(train),

                                    max_steps=1900000

                                    )

 

 

eval_spec = tf.estimator.EvalSpec(input_fn=pandas_train_input_fn(test),

                                    steps=100,

                                    throttle_secs=60

        )

 

 

 

# Model 1: simple linear classifier

#estimator = tf.estimator.LinearClassifier(numeric_featcols+categorical_featcols, n_classes=5)

 

 

estimator = tf.estimator.DNNClassifier(feature_columns=numeric_featcols+indicator_featcols,

                                       config=run_config,

                                       n_classes=5,

                                       dropout=.5,

                                       activation_fn=tf.nn.elu,

                                       hidden_units=[64, 32, 20] #Loss for final step: 24.416494.

                                       #hidden_units=[64, 128, 94, 20, 45, 10]

                                       )

 

 

 

 

 

 

 

#  1,000 = 62.2

#  2,000 = 62.2

# 20,000

 

 

 

 

 

 

 

 

# Train the model

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

   

    

#estimator.train(pandas_train_input_fn(train), steps=1000)

 

 

# Evaluate the model

#eval_result = estimator.evaluate(input_fn=pandas_train_input_fn(test))

#print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

 

#print("The End!!!!!!!!!!!!!")

 

 

# C:\Users\VHABHSGANASS\AppData\Local\Continuum\anaconda3\Scripts\tensorboard.exe --logdir C:\Spiros_Local_Files\TF_Checkpoints\

 

 
