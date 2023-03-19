from PIL import Image
import torchvision as tv
from models import *

def load_img(img_path):
    img = Image.open(img_path)

    transform = tv.transforms.Compose([
        tv.transforms.Resize((32, 32)),
        tv.transforms.ToTensor()
    ])

    img = transform(img)
    return img

model_path = "../models/network_10_train.pth"
img_path = "../imgs/keyboard.jpg"
model = load_model(model_path, map_location=torch.device('cpu'))
img = load_img(img_path)
img = torch.reshape(img, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    outputs = model(img)
    outputs_index = torch.argmax(outputs)
    print(outputs_index)
