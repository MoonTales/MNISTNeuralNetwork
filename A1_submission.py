# Imports
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

def logistic_regression(device):
    # Create our classifier model
    class LogisticRegressionModel(nn.Module):

        class Hyperparameters:
            """
            Hyperparameters:
                learning_rate = 2.3e-2 # How much we update the weights of our model during training
                lambda_value = 0.1 # Regularization strength
                n_epochs = 21 # How many times we pass over the complete training data during training
                log_interval = 75 # How often we print an update during training
            """
            def __init__(self, learning_rate=2.3e-2, lambda_value=0.001, n_epochs=10, log_interval=150):
                self.learning_rate = learning_rate
                self.lambda_value = lambda_value
                self.n_epochs = n_epochs
                self.log_interval = log_interval

        def __init__(self):
            super(LogisticRegressionModel, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

            # -- Create our Other variables -- #
            self.Hyperparameters = self.Hyperparameters()
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            self.train_loader, self.validation_loader, self.test_loader = self.create_Dataloaders(batch_size=64, val_size=12000, transform=transform)
            self.Optimizer = torch.optim.SGD(self.parameters(), lr=self.Hyperparameters.learning_rate, momentum=0.9)
            # -------------------------------- #

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return F.softmax(x, dim=1)

        def create_Dataloaders(self, batch_size=64, val_size=12000, transform=None):
            # Define transformations [Which was taken from the provided notebook]
            # Load MNIST dataset
            full_train_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                                        transform=transform)
            test_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=False, download=True, transform=transform)
            # --

            # Determine the size of the training set
            num_train_samples = len(full_train_set)
            # --

            # Calculate indices for splitting
            train_indices = list(range(num_train_samples))
            val_indices = train_indices[-val_size:]  # Last `val_size` samples for validation
            train_indices = train_indices[:-val_size]  # Rest of the samples for training
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

        def train_model(self, epoch, data_loader, model, optimizer, lambda_value):
            model.train()  # Set model to training mode
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                # L2 regularization
                loss += lambda_value * torch.norm(model.fc.weight, 2)

                loss.backward()
                optimizer.step()
                if batch_idx % self.Hyperparameters.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                               100. * batch_idx / len(data_loader), loss.item()))

        def evaluate(self, data_loader, model, dataset, print_to_log=True, return_accuracy=False):
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
            accuracy = 100. * correct / len(data_loader.dataset)

            if print_to_log:
                print(f'{dataset} set: Avg. loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.0f}%)\n')

            if return_accuracy:
                return accuracy

    LRM = LogisticRegressionModel().to(device)

    LRM.evaluate(LRM.validation_loader, LRM, "Validation")
    for epoch in range(1, LRM.Hyperparameters.n_epochs + 1):
        LRM.train_model(epoch, LRM.train_loader, LRM, LRM.Optimizer, LRM.Hyperparameters.lambda_value)
        LRM.evaluate(LRM.validation_loader, LRM, "Validation")

    results = {}
    results['model'] = LRM
    return results

class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes

        # Define the layers
        self.fc1 = nn.Linear(3072, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer
        self.fc3 = nn.Linear(32, num_classes)  # Output layer

    def forward(self, x):
        # Flatten the input tensor (N x 3 x 32 x 32) to (N x 784)
        x = x.view(-1, 3072)

        # Forward pass through the network
        x = torch.tanh(self.fc1(x))  # Tanh activation after first layer
        x = F.relu(self.fc2(x))  # ReLU activation after second layer
        x = self.fc3(x)  # Output layer (logits)

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

def tune_hyper_parameter(target_metric, device):
    # Logistic Regression Model redefined
    class LogisticRegressionModel(nn.Module):

        class Hyperparameters:
            """
            Hyperparameters:
                learning_rate = 2.3e-2 # How much we update the weights of our model during training
                lambda_value = 0.1 # Regularization strength
                n_epochs = 21 # How many times we pass over the complete training data during training
                log_interval = 75 # How often we print an update during training
            """
            def __init__(self, learning_rate=2.3e-2, lambda_value=0.001, n_epochs=5, log_interval=150):
                self.learning_rate = learning_rate
                self.lambda_value = lambda_value
                self.n_epochs = n_epochs
                self.log_interval = log_interval

        def __init__(self):
            super(LogisticRegressionModel, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

            # -- Create our Other variables -- #
            self.Hyperparameters = self.Hyperparameters()
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            self.train_loader, self.validation_loader, self.test_loader = self.create_Dataloaders(batch_size=64, val_size=12000, transform=transform)
            self.Optimizer = torch.optim.SGD(self.parameters(), lr=self.Hyperparameters.learning_rate, momentum=0.9)
            # -------------------------------- #

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return F.softmax(x, dim=1)

        def create_Dataloaders(self, batch_size=64, val_size=12000, transform=None):
            # Define transformations [Which was taken from the provided notebook]
            # Load MNIST dataset
            full_train_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                                        transform=transform)
            test_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=False, download=True, transform=transform)
            # --

            # Determine the size of the training set
            num_train_samples = len(full_train_set)
            # --

            # Calculate indices for splitting
            train_indices = list(range(num_train_samples))
            val_indices = train_indices[-val_size:]  # Last `val_size` samples for validation
            train_indices = train_indices[:-val_size]  # Rest of the samples for training
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

        def train_model(self, epoch, data_loader, model, optimizer, lambda_value):
            model.train()  # Set model to training mode
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                # L2 regularization
                loss += lambda_value * torch.norm(model.fc.weight, 2)

                loss.backward()
                optimizer.step()
                if batch_idx % self.Hyperparameters.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                               100. * batch_idx / len(data_loader), loss.item()))

        def evaluate(self, data_loader, model, dataset, print_to_log=True, return_accuracy=False):
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
            accuracy = 100. * correct.item() / len(data_loader.dataset)  # Updated line
        
            if print_to_log:
                print(f'{dataset} set: Avg. loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.0f}%)\n')
        
            if return_accuracy:
                return accuracy

    # New Code for Learning Rate
    Learning_rates_to_try = [3.2e-2, 3.23e-2, 3.24e-2, 3.25e-2]
    num_epochs = 8  # Number of epochs to train for
    best_accuracy_lr = 0.0
    best_learning_rate = None
    # Loop through all of the learning rates
    for lr in Learning_rates_to_try:
        # Create our model
        model = LogisticRegressionModel().to(device)
        # Create our optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train the model
        for epoch in range(1, num_epochs + 1):
            model.train_model(epoch, model.train_loader, model, optimizer, model.Hyperparameters.lambda_value)
        
        # Evaluate the model
        validation_accuracy = model.evaluate(model.validation_loader, model, "Validation", False, True)
        
        # Compare with the best accuracy
        if validation_accuracy > best_accuracy_lr:
            best_accuracy_lr = validation_accuracy
            best_learning_rate = lr
    print("Best Learning Rate [Logistic]: ", best_learning_rate)
    
    

    # Second round: Tuning lambda [using the best learning rate found from the first round]
    Lambda_values_to_try = [0.001, 0.01, 0.1]  # Lambda values to try
    best_accuracy_lam = 0.0
    best_lambda = None

    for lam in Lambda_values_to_try:
        # Create our model
        model = LogisticRegressionModel().to(device)
        # Create our optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)
        
        # Train the model
        for epoch in range(num_epochs + 1):  # Use base_epochs_log = 3
            model.train_model(epoch, model.train_loader, model, optimizer, lam)

        # Evaluate the model
        validation_accuracy = model.evaluate(model.validation_loader, model, "Validation", False, True)

        # Compare with the best accuracy
        if validation_accuracy > best_accuracy_lam:
            best_accuracy_lam = validation_accuracy
            best_lambda = lam

    print("Best Lambda Value for Logistic Regression:", best_lambda)

    # Store the results in dictionaries for Logistic Regression
    best_accuracy_log = max(best_accuracy_lr, best_accuracy_lam)
    logistic_params = {
        "Epochs": num_epochs,
        "Learning_rate": best_learning_rate,
        "lambda": best_lambda,
        "validation_accuracy": best_accuracy_log
    }

    # FNN Model
    
    
    
    
    
    # return our best parameters and metrics
    best_params = [logistic_params]
    best_metric = [{"logistic_validation_accuracy": best_accuracy_log}]

    print("Finished tuning hyperparameters")
    return best_params, best_metric
