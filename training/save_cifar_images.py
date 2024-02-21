# This script saves a few images from the CIFAR-10 dataset to a specified folder. These images are used for testing the model's predictions.

import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image

save_folder = 'cifar_images'
os.makedirs(save_folder, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for i in range(10):
    image, label = trainset[i]
    # Convert tensor to PIL image for saving
    img = transforms.ToPILImage()(image)
    img.save(os.path.join(save_folder, f'cifar_sample_{classes[label]}_{i}.png'))
