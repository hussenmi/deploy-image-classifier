import os
import torch
import torch.nn as nn
import torch.optim as optim

def train_network(net, trainloader, model_path='model_weights/cifar10_cnn.pth', epochs=10, lr=0.001, momentum=0.9, optimizer_choice='SGD', force_train=False):
    """
    Train the neural network model.

    Args:
        net (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): The data loader for training data.
        model_path (str, optional): The path to save the trained model. Defaults to 'model_weights/cifar10_cnn.pth'.
        epochs (int, optional): The number of epochs to train the model. Defaults to 10.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        momentum (float, optional): The momentum for the optimizer. Defaults to 0.9.
        optimizer_choice (str, optional): The choice of optimizer. Can be 'SGD' or 'Adam'. Defaults to 'SGD'.
        force_train (bool, optional): Whether to force training even if a trained model exists. Defaults to False.
    """
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    elif optimizer_choice == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Check if a trained model exists and load it
    if os.path.isfile(model_path) and not force_train:
        print(f"Loading saved model from {model_path}")
        net.load_state_dict(torch.load(model_path))
    else:
        print("Training model...")
        total_batches = len(trainloader)
        reporting_interval = max(1, total_batches // 20)  # Report every 5% of the way through, but at least once.

        for epoch in range(epochs):  
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) # Clip the gradients to prevent them from exploding
                optimizer.step()
                
                running_loss += loss.item()
                if i % reporting_interval == reporting_interval - 1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / reporting_interval:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        # Save the trained model for our best model
        # torch.save(net.state_dict(), model_path)