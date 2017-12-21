# Spiro Ganas
# 12/21/17
#
# Run the TensorFlow DNN multiple times to determine the optimal network architecture




import math
import time
import operator
import os
import tensorflow as tf

import preprocess_heart_disease_data
from TensorFlow_DNN_Heart_Disease import MyDNN
from sklearn_heart_disease_models import hidden_layer_generator




if __name__ == "__main__":

    # Turn off tensorflow warnings
    import logging
    logging.getLogger('tensorflow').disabled = True


    start = time.time()

    trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary = preprocess_heart_disease_data.Preprocess_Heart_Disease_Data()

    # Use this if you want to manually create bunch of networks to try
#    MyNetworks = (
#        [10, 5],
#        [20, 5],
#        [30, 5]
#    )





    # or use this if you want to randomly generate networks
    NUMBER_OF_NETWORKS_TO_TEST = 1000
    MyNetworks = hidden_layer_generator(number_of_tuples=NUMBER_OF_NETWORKS_TO_TEST,
                                        max_layers=20,
                                        max_nodes_per_layer=50,
                                        min_nodes_per_layer=5)

    MyDict = {}
    progress = 0
    bestNet = ""
    for net in MyNetworks:
        MyDict[str(net)] = MyDNN(trainX_fully_preprocessed,
                                 trainY_binary,
                                 testX_fully_preprocessed,
                                 testY_binary,
                                 hidden_units=net,
                                 dropout=0.5,
                                 activation_fn=tf.nn.elu,
                                 steps=10_000)
        progress+=1
        if MyDict[str(net)]==max(MyDict.values()):
            bestNet = str(net)
        print("progress: {}. Current Max: {:.3f}".format(progress, max(MyDict.values())))
        print("Best Network so far: ", bestNet)



    from pprint import pprint
    pprint(sorted(MyDict.items(), key=operator.itemgetter(1), reverse=True ))

    #print(MyDict)




    "the code you want to test stays here"
    end = time.time()
    print()
    print("Execution Time: " , math.floor((end - start)/60), " Minutes, ", int((end - start)%60) , " Seconds.")




# RandomForestClassifier F1 Score: 0.861
# MLPClassifier hidden_layer_sizes=(30, 6, 43, 8, 16, 24, 44, 25, 45, 39, 13, 37) alpha = 0.25 F1 Score: 0.857

# TensorFloww DNN Results
# (38, 31, 31, 22, 29, 13)', 0.88412017167381973)
#
#
#  # This is the best model I found so far:
# ('(49, 45, 47, 33, 29, 10, 44, 8, 14, 49, 26, 35, 16, 45, 38)',
#   0.89270386266094426),
#
#  ('(40, 40, 8)', 0.89177489177489166),
#  ('(9, 36, 18, 32, 15, 47, 47, 13, 48, 8)', 0.88986784140969166),
#  ('(18, 20, 11, 20, 41, 24, 29, 36, 21, 36, 10, 9, 15, 18, 42, 33, 24, 17)',
#   0.88986784140969166),
#  ('(7, 16, 11, 20, 8, 14, 39, 8, 35, 36, 44, 33, 14, 16, 48, 30, 20)',
#   0.88888888888888895),
#  ('(34, 7, 50, 41, 16, 11, 23, 14, 8, 11, 24)', 0.88888888888888895),
#  ('(26, 36, 22, 37, 44, 14, 18)', 0.88793103448275879),
#  ('(16,)', 0.88793103448275879),
#  ('(11, 38, 24, 38, 45, 28, 17)', 0.88596491228070184),
#  ('(24, 15, 24, 36, 46, 37, 22, 28, 33, 22, 14, 19, 49, 22, 28, 26)',
#   0.88596491228070184),
#  ('(38, 6, 21, 25, 16, 14, 48, 50, 50, 24, 45, 14, 35)', 0.88510638297872346),
#  ('(46, 46, 41, 35, 6, 33, 13, 22, 16, 50, 41, 35)', 0.88510638297872346),
#  ('(16, 40, 50, 41, 24, 36, 14)', 0.88510638297872346),
#  ('(32, 25, 33, 21, 18, 42, 27, 18, 8, 25, 15, 16, 46, 45, 38, 47, 6)',
#   0.88510638297872346),
#
#
#
#
#
#
#



