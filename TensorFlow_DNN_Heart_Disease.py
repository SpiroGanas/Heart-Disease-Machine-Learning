# Spiro Ganas
# 12/15/17
#
# TensorFlow is Googles machine learning library.
# This script builds a Deep Neural Network using TensorFlow.
# The model should be "better" that the sklearn DNN model,
# because TensorFlow supports dropout regulatization, which
# is a very effective method of preventing the model from
# overfitting the training data.




import tensorflow as tf
from sklearn.metrics import  f1_score

import preprocess_heart_disease_data
from analyze_heart_disease_data import print_results


trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary = preprocess_heart_disease_data.Preprocess_Heart_Disease_Data()






import time
start = time.time()






# a model built using the high-level TFLearn API
Model_01 = tf.contrib.learn.DNNClassifier(hidden_units = [20, 15],
                                          n_classes=2,
                                          dropout=0.5,
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(trainX_fully_preprocessed)
                                          )
#This line makes the tf model compatible with the sklearn way of codding
Model_01 = tf.contrib.learn.SKCompat(Model_01)
Model_01.fit(trainX_fully_preprocessed,
             trainY_binary,
             batch_size=100,
             steps=20000
             )


test_predict_01 = Model_01.predict(testX_fully_preprocessed)['classes']  #The TensorFlow predict object is a bit different from the sklearn object
print_results("TensorFlow DNN", testY_binary, test_predict_01)



f1 = f1_score(testY_binary, test_predict_01)
print("TensorFlow DNN F1 Score: {:.3f}".format(f1))



import math

"the code you want to test stays here"
end = time.time()
print()
print("Execution Time: " , math.floor((end - start)/60), " Minutes, ", int((end - start)%60) , " Seconds.")











