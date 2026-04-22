import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from src.model import CNN

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_epochs, lr, momentum):
        self.net = CNN()
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        self.net.train()
        for _ in range(self.num_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        # Calculate params to send back
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = 0.0, 0.0
        
        self.net.eval()
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                accuracy += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                
        accuracy = accuracy / len(self.valloader.dataset)
        loss = loss / len(self.valloader.dataset)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
