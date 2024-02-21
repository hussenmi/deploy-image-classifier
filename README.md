# Image Classification and Deployment

## Project Overview

A video explanation of the project can be found [here](https://screenapp.io/app/#/shared/9011831a-095f-44e3-8c7a-9e27322f6c5c).

This project is focused on training different convolutional neural network (CNN) models for the task of image classification. We utilize MLFlow for logging, tracking experiments, comparing different model architectures, and ultimately selecting the best-performing model.

## Models Used
- **SimpleCNN**: A straightforward convolutional neural network architecture designed as a baseline for the image classification task.
- **CNNModel2**: A more complex architecture that incorporates batch normalization and dropout layers. This design choice aims to reduce the risk of overfitting, allowing for prolonged training phases and potentially better generalization on unseen data.

## Experimentation and Model Selection
Throughout the training process, model parameters, different metrics, and artifacts were logged to MLFlow. This allows for comparison of models later on. Key metrics such as accuracy, precision, recall, and F1 score were monitored. The best-performing model, as determined by these metrics, was saved for future use.

## How to Run the Training
The project has two main folders -- `training` and `serving`. The `training` folder contains all the necessary files required for training the model. This also includes saving the best model for future use. The project uses different models and hyperparamter combinations for training. To train the models and log to MLFlow, you can use the following command inside the `training` folder:

```
python main.py
```

## How to Use the Deployment

In order to get the deployment up and running, we use the `serving` folder. The `save_cifar_images.py` file just creates some image files for testing. To bring up the server, from the root directory, use the following commands:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python -m serving.app
```

To test the application, you can use two different approaches.

### 1. Web App Interface
Users can interact with a web-based interface to upload images and receive classification predictions directly in the browser. Navigate to the provided URL (e.g., `http://localhost:5000`) and use the upload form to select an image file. Example images are found in the `cifar_images` folder. Upon submission, the predicted class will be displayed on the page alongside the uploaded image.

### 2. API Endpoint
For programmatic access, users can send images via a `curl` command to an API endpoint and receive predictions in a JSON response format. You can follow these commands:

```
cd serving
./curl.sh
```

Make sure to run `chmod +x curl.sh` to make the file executable.
