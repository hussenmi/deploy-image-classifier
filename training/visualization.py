import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler

def imshow(img):
    """
    Display an image using matplotlib.

    Args:
        img (torch.Tensor): The image tensor to be displayed.
    """
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def visualize_predictions(net, dataset, classes, num_images=4):
    """
    Visualizes the predictions made by a neural network on a given dataset.

    Args:
        net (torch.nn.Module): The neural network model.
        dataset (torch.utils.data.Dataset): The dataset to visualize predictions on.
        classes (list): A list of class labels.
        num_images (int, optional): The number of images to visualize. Defaults to 4.
    """
    net.eval()  # Ensure the model is in evaluation mode
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_images, shuffle=True)
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    actual_num_images = images.shape[0]
    # print(f"Batch size: {actual_num_images}, Predictions: {predicted.shape[0]}")

    print('True labels: ', ' '.join(f'{classes[labels[j]]}' for j in range(actual_num_images)))
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(actual_num_images)))

    imshow(torchvision.utils.make_grid(images[:actual_num_images]))
    