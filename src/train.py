import torch
import torch.nn as nn
from network import Network
from datasets import Datasets
from dataloader import Dataloader
from tensorboardX import SummaryWriter
import time

GPU_FLAG = 1 if torch.cuda.is_available() else 0
device = torch.device("cuda:0") if GPU_FLAG else torch.device("cpu")

# datasets
datasets = Datasets()
train_set = datasets['train']
test_set = datasets['test']

train_len = len(train_set)
test_len = len(test_set)

# dataloader
dataloader = Dataloader(datasets, batch_size=64)
train_dataloader = dataloader['train']
test_dataloader = dataloader['test']

# network
network = Network()

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
lr = 1e-3
optimizer = torch.optim.SGD(network.parameters(), lr=lr)

# if GPU_FLAG:
#     network.cuda()
#     loss_fn.cuda()

network.to(device)
loss_fn.to(device)

# training configs
train_it, test_it = 0, 0
epochs = 10

# tensorboard
writer = SummaryWriter("../logs/run")

start_time = time.time()
for epoch in range(epochs):
    # train
    network.train() # not necessary
    print(f"---------Traing at epoch {epoch+1}---------")
    for imgs, targets in train_dataloader:
        # if GPU_FLAG:
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs.to(device)
        targets.to(device)
        outputs = network(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_it += 1
        if train_it % 100 == 0:
            end_time = time.time()
            print(f"Training time: {end_time - start_time}")
            print(f"Training at iteration {train_it}, loss: {loss.item()}")
            writer.add_scalar("Train Loss", loss.item(), train_it)

    # evaluate
    network.eval() # not necessary
    test_loss = 0.
    total_accuracy = 0.
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            # if GPU_FLAG:
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs.to(device)
            targets.to(device)
            outputs = network(imgs)
            loss = loss_fn(outputs, targets)
            test_loss += loss
            accurate_sample = (outputs.argmax(1) == targets).sum()
            total_accuracy += accurate_sample / test_len
    print(f"Loss in the test set: {test_loss}")
    print(f"Accuracy in the test set: {total_accuracy}")
    test_it += 1
    writer.add_scalar("Test Loss", test_loss, test_it)
    writer.add_scalar("Test Accuray", total_accuracy, test_it)

    if (epoch + 1) % 5 == 0:
        torch.save(network, f"../models/network_{epoch+1}_train.pth")

writer.close()
