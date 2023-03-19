import torch
import torch.nn as nn

inputs = torch.tensor([1., 2., 3., 7., 4.])
targets = torch.tensor([1., 2., 5., 6., 0.])

inputs = torch.flatten(inputs)
targets = torch.flatten(targets)

loss = {
    'l1_loss': nn.L1Loss(),
    'mse_loss': nn.MSELoss(reduction='sum'),
    'cross_loss': nn.CrossEntropyLoss(reduction='mean')
}
loss_value = loss['cross_loss'](inputs, targets)
