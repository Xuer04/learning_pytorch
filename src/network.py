import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# Simple template (CIFAR 10 model)
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    network = Network()
    x = torch.ones((64, 3, 32, 32))
    y = network(x)
    # print(y.shape) # torch.Size([64, 10])

    writer = SummaryWriter("../logs/network")
    writer.add_graph(network, x)
    writer.close()


if __name__ == '__main__':
    main()
