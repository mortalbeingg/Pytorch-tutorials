import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math 
import torchvision
import torch.nn.functional as F


class WineDataset(Dataset):
    def __init__(self, transform = None):
        # data loading
        xy = np.loadtxt(
            "./Downloads/wine.csv", delimiter=",", dtype=np.float32, skiprows=1
        )
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        # for dataset indexing
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
            
    def __len__(self):
        return self.n_samples


class ToTensor():
    def __call__(self,sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
        
class Multransform():
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self,sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs,targets

dataset = WineDataset(transform = None)
firstdata = dataset[0]
features, labels = firstdata
print(features)

composed = torchvision.transforms.Compose([ToTensor(),Multransform(2)])

dataset = WineDataset(transform = composed)
firstdata = dataset[0]
features, labels = firstdata
print(type(features))
print(features)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = np.array([2.0,1.0, 0.1])
outputs = softmax(x)
print(f'the output is: {outputs}')

x= torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x,dim=0)
print(outputs)

def cross_entropy(actual, predicted):
    loss = - np.sum(actual*np.log(predicted))
    return loss


Y = np.array([1,0,0])

ypred_1 = np.array([0.7,0.2,0.1])
ypred_2 = np.array([0.1,0.3,0.6])

l1crossentropy =  cross_entropy(Y, ypred_1)
l2crossentropy = cross_entropy(Y, ypred_2)


loss = nn.CrossEntropyLoss()

Y = torch.tensor([2,0,1])
ypred1 = torch.tensor([[0.3,1.0,2.0],[2.0,1.0,0.1],[0.2,2.0,0.5]])
ypred2 = torch.tensor([[2.0,1.0,0.4],[0.5,2.0,0.1],[2.0,0.1,0.5]])

l1 = loss(ypred1, Y)
l2 = loss(ypred2, Y)
torch.argmax(ypred1,1)
torch.argmax(ypred2,1)



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = F.relu()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
        





