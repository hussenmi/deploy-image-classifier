import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batch_size, data_dir='./data'):
    """
    Load the CIFAR-10 dataset and return the trainset, trainloader, testset, and testloader.

    Args:
        batch_size (int): The number of samples per batch.
        data_dir (str, optional): The directory to save the dataset. Defaults to './data'.

    Returns:
        tuple: A tuple containing the trainset, trainloader, testset, and testloader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainset, trainloader, testset, testloader