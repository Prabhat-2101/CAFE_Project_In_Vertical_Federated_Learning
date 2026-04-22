import torch
import torch.nn as nn
from collections import OrderedDict
from src.model import CNN
import flwr as fl

def get_evaluate_fn(testloader):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CNN()
        
        # Set parameters
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)
        
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = 0.0, 0.0
        
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                accuracy += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                
        accuracy = accuracy / len(testloader.dataset)
        loss = loss / len(testloader.dataset)
        return loss, {"accuracy": accuracy}
    return evaluate