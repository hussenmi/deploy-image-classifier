import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_all_predictions(model, data_loader):
    """
    Get the predictions for all the data samples using the given model.

    Args:
        model (torch.nn.Module): The model used for prediction.
        data_loader (torch.utils.data.DataLoader): The data loader containing the samples.

    Returns:
        tuple: A tuple containing two lists - true_labels and predicted_labels.
            true_labels (list): The true labels of the data samples.
            predicted_labels (list): The predicted labels of the data samples.
    """
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())  # Ensure compatibility with CPU
            predicted_labels.extend(predicted.cpu().numpy())  # Ensure compatibility with CPU
    return true_labels, predicted_labels

def get_accuracy(true_labels, predicted_labels):
    """Calculates and returns the accuracy."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def print_classification_metrics(true_labels, predicted_labels):
    print(classification_report(true_labels, predicted_labels))
    

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate various evaluation metrics based on the true labels and predicted labels.

    Args:
        true_labels (array-like): The true labels.
        predicted_labels (array-like): The predicted labels.

    Returns:
        dict: A dictionary containing the evaluation metrics.
            - accuracy (float): The accuracy of the predictions.
            - precision (float): The precision of the predictions.
            - recall (float): The recall of the predictions.
            - f1 (float): The F1 score of the predictions.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    return metrics

    
def plot_confusion_matrix(true_labels, predicted_labels, classes, filename='confusion_matrices/confusion_matrix.png'):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate confusion matrix plot
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()
    plt.close()
