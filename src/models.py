import torch
import torchvision as tv
import torch.nn as nn


def read_model():
    # load network model from torch
    # use `weights` parameter to load pretrained model
    vgg16 = tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT)

    # add module to the network
    # vgg16.add_module('add_linear', nn.Linear(1000, 10))
    vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))

    # change the network architecture
    vgg16.classifier[6] = nn.Linear(4096, 10)

    print(vgg16)

    return vgg16


# method 1: save both network architecture and parameters
def save_model(model, model_path):
    torch.save(model, model_path)


def load_model(model_path, map_location=None):
    return torch.load(model_path, map_location=map_location)


# method 2: save only network parameters
def save_model_dict(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model_dict(model, model_path, map_location=None):
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    # vgg16 = read_model()
    # print(vgg16)
    pass
