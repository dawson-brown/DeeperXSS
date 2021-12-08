import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_precision(predicted_labels: np.array, actual_labels: np.array):
    return precision_score(actual_labels, predicted_labels)


def get_recall(predicted_labels: np.array, actual_labels: np.array):
    return recall_score(actual_labels, predicted_labels)


def get_f1(predicted_labels: np.array, actual_labels: np.array):
    return f1_score(actual_labels, predicted_labels)

def get_accuracy(predicted_labels: np.array, actual_labels: np.array):
    return accuracy_score(actual_labels, predicted_labels)
