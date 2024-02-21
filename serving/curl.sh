#!/bin/bash

# Replace with the appropriate URL of your Flask app
URL="http://localhost:5000/predict"

# Replace with the path to the image you want to upload
IMAGE_PATH="cifar_images/cifar_sample_ship_8.png"

# Perform the HTTP POST request
curl -X POST -F "image=@$IMAGE_PATH" $URL