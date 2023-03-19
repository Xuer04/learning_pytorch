from datasets import Datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

class Dataloader():
    def __init__(self, datasets, batch_size=64):
        self.datasets = datasets
        self.batch_size = batch_size
        self.train_loader = DataLoader(self.datasets['train'], self.batch_size)
        self.test_loader = DataLoader(self.datasets['test'], self.batch_size)
        self.data_loader = {'train': self.train_loader, 'test': self.test_loader}

    def __getitem__(self, index):
        return self.data_loader[index]


datasets = Datasets()
test_set = datasets['test']

test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# img, target = test_set[0] # call __getitem__() method
# print(img.shape) # torch.Size([3, 32, 32])
# print(target) # 3

writer = SummaryWriter("../logs/dataloader")
for epoch in range(2):
    step = 0
    for imgs, targets in test_loader:
        # print(imgs.shape) # torch.Size([64, 3, 32, 32])
        writer.add_images(f"Epoch: {epoch}", imgs, step)
    step += 1

writer.close()
