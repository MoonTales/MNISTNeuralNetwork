"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




### -- Logistic Regression -- ###


def logistic_regression(device):
    learning_rate = 2.3e-2   #2.3e-2
    log_interval = 75 #75
    n_epochs = 21 #21
    results = dict(
        model=None
    )
    # Call our function to get the data loaders
    train_loader, validation_loader, test_loader = create_train_val_loaders()
    # Create our classifier model
    class MultipleLinearRegression(nn.Module):
        def __init__(self):
            super(MultipleLinearRegression, self).__init__()
            self.fc = nn.Linear(28*28, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return F.softmax(x,dim=1)

    multi_linear_model = MultipleLinearRegression().to(device)

    # Use SGD optimizer
    optimizer = torch.optim.SGD(multi_linear_model.parameters(), lr=learning_rate, momentum=0.9)

    def train(epoch, data_loader, model, optimizer):
        model.train()  # Set model to training mode
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            # attempt L2 regularization
            loss += 0.001 * torch.norm(model.fc.weight, 2) #0.01

            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))

    def eval(data_loader, model, dataset):
        model.eval()  # Set model to evaluation mode
        loss = 0
        correct = 0
        with torch.no_grad():  # Disable gradient calculation
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss += F.cross_entropy(output, target, reduction='sum').item()
        loss /= len(data_loader.dataset)
        print(dataset + 'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

    eval(validation_loader, multi_linear_model, "Validation")
    for epoch in range(1, n_epochs + 1):
        train(epoch, train_loader, multi_linear_model, optimizer)
        eval(validation_loader, multi_linear_model, "Validation")
    eval(test_loader, multi_linear_model, "Test")

    results['model'] = multi_linear_model
    return results

def create_train_val_loaders(batch_size=64, val_size=12000):
    # Define transformations [Which was taken from the provided notebook]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    # --

    # Load MNIST dataset
    full_train_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=False, download=True, transform=transform)
    # --

    # Determine the size of the training set
    num_train_samples = len(full_train_set)
    # --

    # Calculate indices for splitting
    train_indices = list(range(num_train_samples))
    val_indices = train_indices[-val_size:]  # Last `val_size` samples for validation
    train_indices = train_indices[:-val_size] # Rest of the samples for training
    # --

    # Create subsets for training and validation using the indices we created above
    train_subset = torch.utils.data.Subset(full_train_set, train_indices)
    val_subset = torch.utils.data.Subset(full_train_set, val_indices)
    # --

    # Create DataLoaders for each of our 3 datasets, train, validation, and test
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    # --

    # Return the DataLoaders
    return train_loader, val_loader, test_loader

def create_train_val_loaders_FNN(batch_size):
    '''
    Function Taken from the FNN_main.py
    '''
    import torch
    from torch.utils.data import random_split
    import torchvision
    """

    :param Params.BatchSize batch_size:
    :return:
    """

    CIFAR_training = torchvision.datasets.CIFAR10('.', train=True, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    CIFAR_test_set = torchvision.datasets.CIFAR10('.', train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # create a training and a validation set
    CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

    train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=batch_size.train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(CIFAR_val_set, batch_size=batch_size.val, shuffle= False)

    test_loader = torch.utils.data.DataLoader(CIFAR_test_set,
                                              batch_size=batch_size.test, shuffle= False)

    return train_loader, val_loader, test_loader

class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes

        # Define the layers
        self.fc1 = nn.Linear(3072, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)   # Second hidden layer
        self.fc3 = nn.Linear(32, num_classes)  # Output layer

    def forward(self, x):
        # Flatten the input tensor (N x 3 x 32 x 32) to (N x 784)
        x = x.view(-1, 3072)

        # Forward pass through the network
        x = torch.tanh(self.fc1(x))  # Tanh activation after first layer
        x = F.relu(self.fc2(x))      # ReLU activation after second layer
        x = self.fc3(x)              # Output layer (logits)

        # Apply softmax to the output to get probabilities
        x = F.log_softmax(x, dim=1)  # Use log_softmax for numerical stability

        return x

    def get_loss(self, output, target):
        # Compute the loss using CrossEntropyLoss
        if self.loss_type == 'cross_entropy' or self.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

def tune_hyper_parameter(target_metric, device):
    # Redefine the Models that will be used for testing the hyperparameters
    best_params, best_metric = None, None

    # Logistic Regression Model
    class LogisticRegression(nn.Module):
        def __init__(self):
            super(LogisticRegression, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return F.softmax(x,dim=1)

    # FNN Model
    class FNN(nn.Module):
        def __init__(self, loss_type, num_classes):
            super(FNN, self).__init__()
            self.loss_type = loss_type
            self.num_classes = num_classes
    
            # Define the layers
            self.fc1 = nn.Linear(3072, 64)  # First hidden layer
            self.fc2 = nn.Linear(64, 32)   # Second hidden layer
            self.fc3 = nn.Linear(32, num_classes)  # Output layer
    
        def forward(self, x):
            # Flatten the input tensor (N x 3 x 32 x 32) to (N x 784)
            x = x.view(-1, 3072)
    
            # Forward pass through the network
            x = torch.tanh(self.fc1(x))  # Tanh activation after first layer
            x = F.relu(self.fc2(x))      # ReLU activation after second layer
            x = self.fc3(x)              # Output layer (logits)
    
            # Apply softmax to the output to get probabilities
            x = F.log_softmax(x, dim=1)  # Use log_softmax for numerical stability
    
            return x
    
        def get_loss(self, output, target):
            # Compute the loss using CrossEntropyLoss
            if self.loss_type == 'cross_entropy' or self.loss_type == 'ce':
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
            return loss

    # Initialize DataLoaders
    LG_train_loader, LG_val_loader, LG_test_loader = create_train_val_loaders()
    import FNN_main as fnn
    #TODO: We need to account for the fact that each image can have a different size, which should be found
    #TODO; Check my FNN class and notice thats its not the same as my working one above
    fnn_train_loader, fnn_val_loader, fnn_test_loader = fnn.get_dataloaders(fnn.Params.BatchSize)

    # Training function
    def train(epoch, data_loader, model, optimizer, log_interval, device):
        model.train()  # Set model to training mode
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                      f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Evaluation function
    def eval(data_loader, model, device):
        model.eval()  # Set model to evaluation mode
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(data_loader.dataset) * 100.0
        return accuracy

    # Hyperparameter tuning for Logistic Regression - Learning Rate
    base_learning_rate_log = 1.0e-3  # Default learning rate
    base_lambda_log = 0.1  # Default lambda value
    learning_rates_logistic = [2.2e-2, 2.3e-2, 2.3e-2, 2.5e-2]  # Learning rates to try

    best_accuracy_logistic = 0.0
    best_learning_rate_logistic = None

    # First round: Tuning learning rates for Logistic Regression
    for lr in learning_rates_logistic:
        logistic_regression_model = LogisticRegression().to(device)  # Re-initialize the model
        optimizer = torch.optim.Adam(logistic_regression_model.parameters(), lr=lr)

        for epoch in range(1, 4):  # Use base_epochs_log = 3
            train(epoch, LG_train_loader, logistic_regression_model, optimizer, log_interval=100, device=device)

        validation_accuracy = eval(LG_val_loader, logistic_regression_model, device)

        if validation_accuracy > best_accuracy_logistic:
            best_accuracy_logistic = validation_accuracy
            best_learning_rate_logistic = lr

    print("Best Logistic Regression Learning Rate:", best_learning_rate_logistic)

    # Second round: Tuning lambda using the best learning rate found
    lambda_values = [0.01, 0.1, 1.0, 10.0, 100.0]  # Lambda values to try
    best_accuracy_lambda = 0.0
    best_lambda = None

    for lam in lambda_values:
        logistic_regression_model = LogisticRegression().to(device)  # Re-initialize the model
        optimizer = torch.optim.Adam(logistic_regression_model.parameters(), lr=best_learning_rate_logistic)

        for epoch in range(1, 4):  # Use base_epochs_log = 3
            train(epoch, LG_train_loader, logistic_regression_model, optimizer, log_interval=100, device=device)

        # Here you would need to adjust for how lambda is incorporated into your model, e.g., regularization.
        validation_accuracy = eval(LG_val_loader, logistic_regression_model, device)

        if validation_accuracy > best_accuracy_lambda:
            best_accuracy_lambda = validation_accuracy
            best_lambda = lam

    print("Best Lambda Value for Logistic Regression:", best_lambda)

    # Store the results in dictionaries for Logistic Regression
    logistic_params = {
        "Epochs": 3,
        "Learning_rate": best_learning_rate_logistic,
        "lambda": best_lambda,
        "validation_accuracy": best_accuracy_lambda
    }

    # Hyperparameter tuning for FNN - Learning Rate
    base_learning_rate_fnn = 1.0e-3  # Default learning rate for FNN
    learning_rates_fnn = [base_learning_rate_fnn, 5e-3, 1e-2, 1e-1]  # Learning rates to try

    best_accuracy_fnn = 0.0
    best_learning_rate_fnn = None

    # First round: Tuning learning rates for FNN
    for lr in learning_rates_fnn:
        fnn_model = FNN('ce', 10).to(device)  # Re-initialize the model
        optimizer = torch.optim.Adam(fnn_model.parameters(), lr=lr)

        for epoch in range(1, 4):  # Use base_epochs_log = 3
            train(epoch, fnn_train_loader, fnn_model, optimizer, log_interval=100, device=device)

        validation_accuracy = eval(fnn_val_loader, fnn_model, device)

        if validation_accuracy > best_accuracy_fnn:
            best_accuracy_fnn = validation_accuracy
            best_learning_rate_fnn = lr

    print("Best FNN Learning Rate:", best_learning_rate_fnn)

    # Second round: Tuning lambda for FNN using the best learning rate found
    best_accuracy_lambda_fnn = 0.0
    best_lambda_fnn = None

    for lam in lambda_values:
        fnn_model = FNN('ce', 10).to(device)  # Re-initialize the model
        optimizer = torch.optim.Adam(fnn_model.parameters(), lr=best_learning_rate_fnn)

        for epoch in range(1, 4):  # Use base_epochs_log = 3
            train(epoch, fnn_train_loader, fnn_model, optimizer, log_interval=100, device=device)

        validation_accuracy = eval(fnn_val_loader, fnn_model, device)

        if validation_accuracy > best_accuracy_lambda_fnn:
            best_accuracy_lambda_fnn = validation_accuracy
            best_lambda_fnn = lam

    print("Best Lambda Value for FNN:", best_lambda_fnn)

    # Store the results in dictionaries for FNN
    fnn_params = {
        "Epochs": 3,  # Default epochs used
        "Learning_rate": best_learning_rate_fnn,
        "lambda": best_lambda_fnn,
        "validation_accuracy": best_accuracy_lambda_fnn
    }

    best_params = [logistic_params, fnn_params]
    best_metric = [{"logistic_validation_accuracy": best_accuracy_lambda}, {"fnn_validation_accuracy": best_accuracy_lambda_fnn}]

    print("Finished tuning hyperparameters")
    return best_params, best_metric
