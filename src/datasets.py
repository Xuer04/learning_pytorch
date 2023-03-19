from torchvision.datasets.cifar import CIFAR10
import torchvision as tv

class Datasets():
    def __init__(self):
        dataset_transform = tv.transforms.Compose([
            # tv.transforms.Resize([1024, 1024]),
            tv.transforms.ToTensor()
        ])

        # read datasets CIFAR10
        self.train_set = CIFAR10(root="../datasets", train=True, transform = dataset_transform, download=False)
        self.test_set = CIFAR10(root="../datasets", train=False, transform = dataset_transform, download=False)
        self.datasets = {'train': self.train_set, 'test': self.test_set}

    def __getitem__(self, index):
        return self.datasets[index]
