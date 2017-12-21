# Spiro Ganas
# 12/15/17
#
# TensorFlow is Googles machine learning library.
# This script builds a Deep Neural Network using TensorFlow.
# The model should be "better" that the sklearn DNN model,
# because TensorFlow supports dropout regulatization, which
# is a very effective method of preventing the model from
# overfitting the training data.



import os

import tensorflow as tf
from sklearn.metrics import  f1_score


from analyze_heart_disease_data import print_results, hidden_layer_generator



# Turn off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}













def MyDNN(trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary, hidden_units = [10,5], dropout=0.5,activation_fn=tf.nn.elu, steps=1_000 ):


    # a model built using the high-level TFLearn API
    Model_01 = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units,
                                              n_classes=2,
                                              dropout=dropout,
                                              feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(trainX_fully_preprocessed),
                                              activation_fn=activation_fn
                                              ,gradient_clip_norm=0.9
                                              ,optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                                              )
    #This line makes the tf model compatible with the sklearn way of codding
    Model_01 = tf.contrib.learn.SKCompat(Model_01)
    Model_01.fit(trainX_fully_preprocessed,
                 trainY_binary,
                 batch_size=100,
                 steps=steps
                 )


    test_predict_01 = Model_01.predict(testX_fully_preprocessed)['classes']  #The TensorFlow predict object is a bit different from the sklearn object
    print_results("TensorFlow DNN", testY_binary, test_predict_01)



    f1 = f1_score(testY_binary, test_predict_01)
    print("TensorFlow DNN F1 Score: {:.3f}".format(f1))
    return f1



if __name__ == "__main__":
    import logging
    logging.getLogger('tensorflow').disabled = True

    import random
    random.seed(1)

    import time
    import math
    start = time.time()

    import preprocess_heart_disease_data
    trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary = preprocess_heart_disease_data.Preprocess_Heart_Disease_Data()

    #net = [149, 145, 47, 33, 29, 10, 44, 28, 14, 49, 26, 35, 16, 45, 38]
    net=[10, 10]

    print(net)

    F1 = MyDNN(trainX_fully_preprocessed,
                         trainY_binary,
                         testX_fully_preprocessed,
                         testY_binary,
                         hidden_units=net,
                         dropout=0.5,
                         activation_fn=tf.nn.elu,
                         steps=10_000)
    print('F1 Score: {:.3f}'.format(F1))



    end = time.time()
    print()
    print("Execution Time: " , math.floor((end - start)/60), " Minutes, ", int((end - start)%60) , " Seconds.")












