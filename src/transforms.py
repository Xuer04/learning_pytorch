from torchvision import transforms
from PIL import Image
from tensorboardX import SummaryWriter

writer = SummaryWriter("../logs")

img = Image.open("../imgs/keyboard.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_t = trans_totensor(img)
writer.add_image("ToTensor", img_t)
# print(img_t.shape) torch.Size([3, 4000, 6000])

# Normalize
trans_norm = transforms.Normalize([.5, .5, .5], [.5, .5, .5])
img_norm = trans_norm(img_t)
writer.add_image("Normalize", img_norm)
# print(img_norm) [-1, 1]

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize)
# print(img_resize.shape) torch.Size([3, 512, 512])

# Compose
trans_compoese = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
img_compose = trans_compoese(img)
# print(img_compose.shape) torch.Size([3, 512, 512])
writer.add_image("Compose", img_compose)

writer.close()
