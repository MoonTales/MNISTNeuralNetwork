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
                n_epochs = 21 # How many times we pass over the complete training data during training
                learning_rate = 2.3e-2 # How much we update the weights of our model during training
                weight_decay = 0.001 # Regularization strength
                momentum = 0.9 # Momentum for SGD
                log_interval = 75 # How often we print an update during training
            """
            def __init__(self, n_epochs=10, learning_rate=2.3e-2, weight_decay=0.001, momentum = 0.9, log_interval=150):
                self.n_epochs = n_epochs
                self.learning_rate = learning_rate
                self.weight_decay = weight_decay
                self.momentum = momentum
                self.log_interval = log_interval

            # Setters
            def set_n_epochs(self, n_epochs):
                self.n_epochs = n_epochs

            def set_learning_rate(self, learning_rate):
                self.learning_rate = learning_rate

            def set_weight_decay(self, weight_decay):
                self.weight_decay = weight_decay

            def set_momentum(self, momentum):
                self.momentum = momentum

            def set_log_interval(self, log_interval):
                self.log_interval = log_interval

        def __init__(self):
            super(LogisticRegressionModel, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

            # -- Create our Other variables -- #
            self.Hyperparameters = self.Hyperparameters()
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            self.train_loader, self.validation_loader, self.test_loader = self.create_Dataloaders(batch_size=64,
                                                                                                  val_size=12000,
                                                                                                  transform=transform)
            self.Optimizer = torch.optim.SGD(self.parameters(), lr=self.Hyperparameters.learning_rate,
                                             momentum=self.Hyperparameters.momentum,
                                             weight_decay=self.Hyperparameters.weight_decay)

        def set_hyperparameters(self, n_epochs, learning_rate, weight_decay, momentum, log_interval):
            '''
            Set the hyperparameters for the model, and update the optimizer
            '''
            self.Hyperparameters.set_n_epochs(n_epochs)
            self.Hyperparameters.set_learning_rate(learning_rate)
            self.Hyperparameters.set_weight_decay(weight_decay)
            self.Hyperparameters.set_momentum(momentum)
            self.Hyperparameters.set_log_interval(log_interval)
            self.Optimizer = torch.optim.SGD(self.parameters(), lr=self.Hyperparameters.learning_rate,
                                             momentum=self.Hyperparameters.momentum,
                                             weight_decay=self.Hyperparameters.weight_decay)
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

        def train_model(self, epoch, data_loader, model, optimizer):
            model.train()  # Set model to training mode
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.Hyperparameters.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                               100. * batch_idx / len(data_loader), loss.item()))

        def evaluate(self, data_loader, model, dataset, print_to_log=True, return_accuracy=True):
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
                print(
                    f'{dataset} set: Avg. loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')

            if return_accuracy:
                return accuracy

    LRM = LogisticRegressionModel().to(device)
    LRM.set_hyperparameters(n_epochs=12, learning_rate=0.014, weight_decay=01e-05, momentum=0.95, log_interval=75)
    LRM.evaluate(LRM.validation_loader, LRM, "Validation")
    for epoch in range(1, LRM.Hyperparameters.n_epochs + 1):
        LRM.train_model(epoch, LRM.train_loader, LRM, LRM.Optimizer)
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
        x = F.softmax(x, dim=1)

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

    should_tune_log = True
    should_tune_fnn = True
    best_accuracy_log = 0
    logistic_params = None
    best_accuracy_fnn = 0
    FNN_Params = None
    if should_tune_log:
        # Redefine the LogisticRegressionModel
        class LogisticRegressionModel(nn.Module):

            class Hyperparameters:
                """
                Hyperparameters:
                    n_epochs = 21 # How many times we pass over the complete training data during training
                    learning_rate = 2.3e-2 # How much we update the weights of our model during training
                    weight_decay = 0.001 # Regularization strength
                    momentum = 0.9 # Momentum for SGD
                    log_interval = 75 # How often we print an update during training
                """
                def __init__(self, n_epochs=10, learning_rate=2.3e-2, weight_decay=0.001, momentum = 0.9, log_interval=150):
                    self.n_epochs = n_epochs
                    self.learning_rate = learning_rate
                    self.weight_decay = weight_decay
                    self.momentum = momentum
                    self.log_interval = log_interval

                # Setters
                def set_n_epochs(self, n_epochs):
                    self.n_epochs = n_epochs

                def set_learning_rate(self, learning_rate):
                    self.learning_rate = learning_rate

                def set_weight_decay(self, weight_decay):
                    self.weight_decay = weight_decay

                def set_momentum(self, momentum):
                    self.momentum = momentum

                def set_log_interval(self, log_interval):
                    self.log_interval = log_interval

            def __init__(self):
                super(LogisticRegressionModel, self).__init__()
                self.fc = nn.Linear(28 * 28, 10)

                # -- Create our Other variables -- #
                self.Hyperparameters = self.Hyperparameters()
                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                self.train_loader, self.validation_loader, self.test_loader = self.create_Dataloaders(batch_size=64,
                                                                                                      val_size=12000,
                                                                                                      transform=transform)
                self.Optimizer = torch.optim.Adam(self.parameters(), lr=self.Hyperparameters.learning_rate,
                                                  weight_decay=self.Hyperparameters.weight_decay)

            def set_hyperparameters(self, n_epochs, learning_rate, weight_decay, momentum, log_interval):
                '''
                Set the hyperparameters for the model, and update the optimizer
                '''
                self.Hyperparameters.set_n_epochs(n_epochs)
                self.Hyperparameters.set_learning_rate(learning_rate)
                self.Hyperparameters.set_weight_decay(weight_decay)
                self.Hyperparameters.set_momentum(momentum)
                self.Hyperparameters.set_log_interval(log_interval)
                self.Optimizer = torch.optim.SGD(self.parameters(), lr=self.Hyperparameters.learning_rate,
                                                 momentum=self.Hyperparameters.momentum,
                                                 weight_decay=self.Hyperparameters.weight_decay)
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

            def train_model(self, epoch, data_loader, model, optimizer, show_output=True):
                model.train()  # Set model to training mode
                for batch_idx, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % self.Hyperparameters.log_interval == 0:
                        if show_output:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(data), len(data_loader.dataset),
                                       100. * batch_idx / len(data_loader), loss.item()))

            def evaluate(self, data_loader, model, dataset, print_to_log=True, return_accuracy=True):
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
                    print(
                        f'{dataset} set: Avg. loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.0f}%)\n')

                if return_accuracy:
                    return accuracy
        best_accuracy = 0
        best_params = None
        dynamic_epochs = 5  # Start with a default number of epochs
        max_epochs = 15  # Maximum number of epochs

        # Define the hyperparameters to search over
        # what to submit, lr = 0.016, weight_decay = 1e-05
        '''
        Values attemped from previous runs:
        learning_rates = [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020]
        weight_decays = [0.01, 0.001, 0.0001, 0.00001]
        '''
        learning_rates = [0.016, 0.018]
        weight_decays = [0.0001, 0.00001]

        # Grid search over the hyperparameter combinations
        for lr in learning_rates:
            for wd in weight_decays:
                print(f"-- Testing with lr={lr}, weight_decay={wd} --")
                LRM = LogisticRegressionModel().to(device)
                # Set the hyperparameters for the current test
                LRM.set_hyperparameters(n_epochs=dynamic_epochs,  # Start with the tracked dynamic_epochs
                                        learning_rate=lr,
                                        weight_decay=wd,
                                        momentum=0.95,
                                        log_interval=LRM.Hyperparameters.log_interval)

                # Train and evaluate for each epoch (up to dynamic_epochs)
                for epoch in range(1, dynamic_epochs + 1):
                    LRM.train_model(epoch, LRM.train_loader, LRM, LRM.Optimizer, show_output=False)
                    val_accuracy = LRM.evaluate(LRM.validation_loader, LRM, "Validation", print_to_log=False, return_accuracy=True)
                    # Check if this combination gives a better accuracy
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = (lr, wd)
                        print(f"IMPORTANT - New best accuracy: {best_accuracy:.2f}% with lr={lr}, weight_decay={wd} at epoch [{epoch}]")
                        # continue training for more epochs if the accuracy is still improving
                        dynamic_epochs = epoch + 1
                        # just don't go over the max_epochs
                        if epoch > max_epochs:
                            break

        print(f"Best hyperparameters Logistic: lr={best_params[0]}, weight_decay={best_params[1]} with accuracy {best_accuracy:.2f}%")

        logistic_params = {"learning_rate": best_params[0], "weight_decay": best_params[1]}
        best_accuracy_log = best_accuracy
    if should_tune_fnn:
        # Redefine the FNN model
        class FNN_class(nn.Module):
            def __init__(self, loss_type, num_classes):
                super(FNN_class, self).__init__()
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
                x = F.softmax(x, dim=1)  # Use log_softmax for numerical stability

                return x

            def get_loss(self, output, target):
                # Compute the loss using CrossEntropyLoss
                if self.loss_type == 'cross_entropy' or self.loss_type == 'ce':
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(output, target)
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")

                return loss
        import FNN_main as FNN_main
        class Params:
            class BatchSize:
                train = 128
                val = 128
                test = 1000
        train_loader, val_loader, test_loader = FNN_main.get_dataloaders(Params.BatchSize)
        best_accuracy = 0
        best_params = None
        dynamic_epochs = 5  # Start with a default number of epochs
        max_epochs = 15  # Maximum number of epochs

        # Define the hyperparameters to search over
        # check 0.0005 and 0.0005
        learning_rates = [1e-4, 5e-4]
        weight_decays = [1e-5, 5e-4]
        # Grid search over the hyperparameter combinations

        # New training function
        from tqdm import tqdm
        def train_FNN(net, optimizer, train_loader, device):
            # Function taken from the FNN_main.py file
            net.train()
            pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
            avg_loss = 0
            for batch_idx, (data, target) in enumerate(pbar):
                optimizer.zero_grad()
                data = data.to(device)
                target = target.to(device)
                output = net(data)
                loss = net.get_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_sc = loss.item()

                avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

                pbar.set_description('train loss: {:.6f} avg loss: {:.6f}'.format(loss_sc, avg_loss))

        # New evaluation function
        def validation_FNN(net, validation_loader, device):
            net.eval()
            validation_loss = 0
            correct = 0
            for data, target in validation_loader:
                data = data.to(device)
                target = target.to(device)
                output = net(data)
                loss = net.get_loss(output, target)
                validation_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()

            validation_loss /= len(validation_loader.dataset)
            print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                validation_loss, correct, len(validation_loader.dataset),
                100. * correct / len(validation_loader.dataset)))
            # returns the accuracy
            return 100. * correct / len(validation_loader.dataset)

        for lr in learning_rates:
            for wd in weight_decays:
                print(f"-- Testing with lr={lr}, weight_decay={wd} --")
                # Initialize FNN with the correct arguments
                FNN = FNN_class(loss_type='ce', num_classes=10).to(device)
                optimizer = torch.optim.Adam(FNN.parameters(), lr=lr, weight_decay=wd)
                for epoch in range(1, dynamic_epochs + 1):
                    train_FNN(FNN, optimizer, train_loader, device)
                    val_accuracy = validation_FNN(FNN, val_loader, device)
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = (lr, wd)
                        print(f"IMPORTANT - New best accuracy: {best_accuracy:.2f}% with lr={lr}, weight_decay={wd} at epoch [{epoch}]")
                        dynamic_epochs = epoch + 1
                        if epoch > max_epochs:
                            break

        print(f"Best hyperparameters FNN: lr={best_params[0]}, weight_decay={best_params[1]} with accuracy {best_accuracy:.2f}%")

        FNN_Params = {"learning_rate": best_params[0], "weight_decay": best_params[1]}
        best_accuracy_fnn = best_accuracy
    # Return the best parameters and metrics
    best_params = [logistic_params, FNN_Params]
    best_metric = [{"logistic_validation_accuracy": best_accuracy_log}, {"FNN_validation_accuracy": best_accuracy_fnn}]

    print("Finished tuning hyperparameters")
    return best_params, best_metric