import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_precision(predicted_labels: np.array, actual_labels: np.array):
    precision = precision_score(actual_labels, predicted_labels)
    print("Precision is: ", precision)
    return precision


def get_recall(predicted_labels: np.array, actual_labels: np.array):
    recall = recall_score(actual_labels, predicted_labels)
    print("Recall is: ", recall)
    return recall


def get_f1(predicted_labels: np.array, actual_labels: np.array):
    f1 = f1_score(actual_labels, predicted_labels)
    print("F1 is: ", f1)
    return f1

def get_accuracy(predicted_labels: np.array, actual_labels: np.array):
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print("Accuracy is: ", accuracy)
    return accuracy