
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from random import randint



import preprocess_heart_disease_data



trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary = preprocess_heart_disease_data.Preprocess_Heart_Disease_Data()





def print_results(model_name,testY_binary, test_predict):
    """Prints the confusion matrix and the F1 score of a model."""
    print()
    print("####################################")
    cm = confusion_matrix(testY_binary, test_predict)
    print(model_name + " Confusion Matrix:")
    print(cm)
    print()
    f1 = f1_score(testY_binary, test_predict)
    print(model_name + " F1 Score: {:.3f}".format(f1))
    area_under_curve = roc_auc_score(testY_binary, test_predict)
    print(model_name + " ROC Area Under the Curve: {:.3f}".format(area_under_curve))

    print("####################################")
    print()




def hidden_layer_generator(number_of_tuples = 1000, max_layers = 15, max_nodes_per_layer=50, min_nodes_per_layer=5):
    """Returns a list of tuples containing randomly generated hidden layer tuples.
       This is used as a parameter when applying the RandomizedSearchCV to a MLPClassifier.
    """
    MyTuples = []
    for tuple_counter in range(1 ,number_of_tuples+1):
        number_of_layers = randint(1, max_layers)
        temp = []
        for x in range(1, number_of_layers + 1):
            temp.append(randint(min_nodes_per_layer, max_nodes_per_layer))
        MyTuples.append(tuple(temp))
    return MyTuples




def run_some_simple_models():
    """This function runs a bunch of sklearn models.  Nothing too fancy..."""


    # A dictionary containing the name of the models you want to run and their function definitions
    Models = {"NearestNeighbors": KNeighborsClassifier(n_neighbors =10),
              "SGDClassifier": SGDClassifier(random_state=1),
              "RandomForestClassifier": RandomForestClassifier(n_estimators=1500),
              "AdaBoostClassifier":AdaBoostClassifier(n_estimators=1500),
              "MLPClassifier hidden_layer_sizes=(10)":MLPClassifier(solver='lbfgs',
                                                                    alpha=1e-5,
                                                                    hidden_layer_sizes=(10),
                                                                    random_state=1),
              "MLPClassifier hidden_layer_sizes=(20, 15, 10)": MLPClassifier(solver='lbfgs',
                                                                     alpha=1e-5,
                                                                     hidden_layer_sizes=(20, 15, 10),
                                                                     random_state=1),
              "MLPClassifier hidden_layer_sizes = (30, 6, 43, 8, 16, 24, 44, 25, 45, 39, 13, 37)":MLPClassifier(solver='lbfgs',
                                                                                                                alpha=1e-5,
                                                                                                                hidden_layer_sizes=(30, 6, 43, 8, 16, 24, 44, 25, 45, 39, 13, 37),
                                                                                                                random_state=1),
              "MLPClassifier hidden_layer_sizes=(8, 11, 16, 10, 8, 19, 15, 14, 16, 8, 9, 8, 11, 20, 11, 11, 13, 10) alpha = 0.1": MLPClassifier(solver='lbfgs',
                                                                                                                                                alpha=.1,
                                                                                                                                                hidden_layer_sizes=(8, 11, 16, 10, 8, 19, 15, 14, 16, 8, 9, 8, 11, 20, 11, 11, 13, 10),
                                                                                                                                                random_state=1),
              "MLPClassifier hidden_layer_sizes=(30, 6, 43, 8, 16, 24, 44, 25, 45, 39, 13, 37) alpha = 0.25": MLPClassifier(
                                                                                                                              solver='lbfgs',
                                                                                                                              alpha=.25,
                                                                                                                              hidden_layer_sizes=(30, 6, 43, 8, 16, 24, 44, 25, 45, 39, 13, 37),
                                                                                                                              random_state=20)
              }

    # This fits each model in the dictionary and measures their performance using the F1 score

    best_model = "none"
    best_model_f1 = 0

    for modelName, model in Models.items():
        model.fit(trainX_fully_preprocessed, trainY_binary)
        test_predict = model.predict(testX_fully_preprocessed)
        print_results(modelName, testY_binary, test_predict)
        f1 = f1_score(testY_binary, test_predict)

        if f1>best_model_f1:
            best_model, best_model_f1 = modelName, f1

    print("The best model was the \"{}\" with an F1 score of {:.3f} ".format( best_model, best_model_f1))
    exit()






def run_fancy_model(n_iter=20):
    """"This function is more complex, it develops a deep neural network that uses an architecture and
        Hyperparameters that have been optimized using a random search cross validation technique.

        Args:
        n_iter (int): The number of cross validations run to determine the best hyperparameter.

        """

    # Model_05 - a deep neural network classifier optimized using RandomizedSearchCV

    # Set up the paramter dictionary used by the RandomizedSearchCV
    hidden_layer_sizes_parameters = hidden_layer_generator(number_of_tuples = 1000, max_layers = 50, max_nodes_per_layer=20, min_nodes_per_layer=8)

    MLP_Parameters = {'alpha': [ .3, .2, .1, .05, .01], #1e-6, 1e-5, 1e-4, 1e-3,
                      'hidden_layer_sizes': hidden_layer_sizes_parameters,
                      'solver': ['lbfgs'],
                      'random_state':[20],
                      'max_iter': [500]
                      }

    Model_05 = RandomizedSearchCV(MLPClassifier(),
                                  MLP_Parameters,
                                  n_iter=n_iter,
                                  scoring='f1',
                                  refit=True,
                                  n_jobs =1)

    Model_05.fit(trainX_fully_preprocessed, trainY_binary)
    test_predict_05 = Model_05.predict(testX_fully_preprocessed)

    print("RandomizedSearchCV MLP Classifier Best Results:")
    print("Best CV F1 Score: {:.3f}".format(Model_05.best_score_))
    print("Best Parameters: {}".format(Model_05.best_params_))
    print_results("RandomizedSearchCV MLP Classifier", testY_binary, test_predict_05)


if __name__=="__main__":
    run_some_simple_models()
    #run_fancy_model(n_iter=200)