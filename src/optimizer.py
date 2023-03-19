import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from .network import Network


datasets = CIFAR10("../datasets", train=False, transform=tv.transforms.ToTensor(), download=False)
dataloader = DataLoader(datasets, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

network = Network()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)

for epoch in range(20):
    running_loss = 0.
    for imgs, targets in dataloader:
        # get outputs
        outputs = network(imgs)
        # calculate loss value
        loss_value = loss(outputs, targets)
        # set the gradients to zero
        optimizer.zero_grad()
        # calculate all gradients
        loss_value.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        running_loss += loss_value
    print(running_loss)
