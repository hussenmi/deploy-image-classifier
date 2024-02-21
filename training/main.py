import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifact, log_artifacts
from model import SimpleCNN, CNNModel2
from train import train_network
from data_loading import load_cifar10
from evaluate import calculate_metrics, print_classification_metrics, get_all_predictions, plot_confusion_matrix
from visualization import visualize_predictions
import torch
import random


def run_experiment(model_name, hyperparams):
    """
    Run an experiment with the given model and hyperparameters. It also logs parameters and metrics to MLflow.

    Args:
        model_name (str): The name of the model to use for the experiment.
        hyperparams (dict): A dictionary containing the hyperparameters for the experiment. The hyperparameters are epoch, lr, momentum, batch_size, and optimizer_choice.

    Returns:
        None
    """
    epochs, lr, momentum, batch_size, optimizer_choice = hyperparams.values()
    # print(f"Running experiment with: model_name={model_name}, epochs={epochs}, lr={lr}, momentum={momentum}, batch_size={batch_size}")
    with mlflow.start_run():
        log_params({
            "model_name": model_name,
            "epochs": epochs,
            "lr": lr,
            "momentum": momentum,
            "batch_size": batch_size
        })
        
        if model_name == "SimpleCNN":
            net = SimpleCNN()
        elif model_name == "CNNModel2":
            net = CNNModel2()
            
        trainset, trainloader, testset, testloader = load_cifar10(batch_size)
        
        # Visualize predictions before training (using an untrained model)
        print("Before Training:")
        visualize_predictions(net, testset, classes)

        
        # Training
        model_filename = f"model_weights/{model_name}_lr{lr}_momentum{momentum}_epochs{epochs}.pth"
        train_network(net, trainloader, model_path=model_filename, epochs=epochs, lr=lr, momentum=momentum, optimizer_choice=optimizer_choice, force_train=True)
        net.load_state_dict(torch.load(model_filename))  # Ensure using the trained model

        
        # Visualize predictions after training
        print("After Training:")
        visualize_predictions(net, testset, classes)

        # Evaluation
        true_labels, predicted_labels = get_all_predictions(net, testloader)
        print_classification_metrics(true_labels, predicted_labels)
        # accuracy = get_accuracy(true_labels, predicted_labels)
        metrics = calculate_metrics(true_labels, predicted_labels)

        # plot_confusion_matrix(true_labels, predicted_labels, classes)

        # Log metrics
        for metric_name, value in metrics.items():
            log_metric(metric_name, value)
        
        # Save and log confusion matrix plot with a dynamic filename
        plot_filename = f"confusion_matrices/{model_name}_confusion_matrix_lr{lr}_momentum{momentum}_epochs{epochs}.png"
        plot_confusion_matrix(true_labels, predicted_labels, classes, filename=plot_filename)

        # Log the confusion matrix image as an MLflow artifact
        log_artifact(plot_filename)




if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("CIFAR-10_Experiments")
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Define the hyperparameter search space
    
    hyperparameters_simplecnn = {
        "epochs": [5, 7, 10, 15],
        "lr": [0.001, 0.0005, 0.0001],
        "momentum": [0.9, 0.95, 0.99, 0.999],
        "batch_size": [8, 16, 32, 64, 128],
        "optimizer": ["SGD", "Adam"]
    }

    hyperparameters_cnnmodel2 = {
        "epochs": [10, 15, 25],
        "lr": [0.0005, 0.0001, 0.001, 0.01],
        "momentum": [0.9, 0.95, 0.99, 0.999],
        "batch_size": [16, 32, 64, 128],
        "optimizer": ["SGD", "Adam"]
    }
    
    models_to_test = [('SimpleCNN', hyperparameters_simplecnn), ('CNNModel2', hyperparameters_cnnmodel2)]
    # models_to_test = [('CNNModel2', hyperparameters_cnnmodel2)]
    
    experiments_per_model = 8  # Define how many experiments to run for each model

    for model_name, hyperparams_space in models_to_test:
        # print(model_name, hyperparams_space)
        for _ in range(experiments_per_model):
            # Sample a random set of hyperparameters for each experiment
            hyperparams = {k: random.choice(v) for k, v in hyperparams_space.items()}
            
            print(f"Running experiment with: {model_name}, hyperparams: {hyperparams}")
            
            run_experiment(model_name, hyperparams)
            