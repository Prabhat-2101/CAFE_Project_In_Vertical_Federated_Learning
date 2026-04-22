import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform:
            img = self.transform(img)
        return img, target

def load_data(num_clients, alpha, batch_size):
    """
    Load MNIST and partition it among `num_clients`.
    `alpha` can be a float for dirichlet distribution or 'IID'.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)
    
    if alpha == "IID":
        # IID Partitioning
        partition_size = len(trainset) // num_clients
        indices = np.random.permutation(len(trainset))
        client_trainloaders = []
        for i in range(num_clients):
            idx = indices[i * partition_size:(i + 1) * partition_size]
            subset = torch.utils.data.Subset(trainset, idx)
            client_trainloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    else:
        # Dirichlet Partitioning
        alpha = float(alpha)
        num_classes = 10
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(trainset.targets)
        y_train = np.array(trainset.targets)
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                
        client_trainloaders = []
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            subset = torch.utils.data.Subset(trainset, idx_batch[j])
            client_trainloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
            
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return client_trainloaders, testloader
