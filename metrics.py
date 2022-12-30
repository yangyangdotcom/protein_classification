# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import averange_predictions_score
import numpy as np
import pandas as pd
from math import sqrt

def get_accuracy(actual, predicted):
    correct = 0

    for i in range(len(predicted)):
        if np.array_equal(predicted[i], actual[i]):
            correct += 1
    return correct / float(len(predicted))

# def get_tp(actual, predicted):
