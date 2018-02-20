from PIL import Image
from torchvision import transforms as T

path = '/home/ubuntu/Pictures/000001.jpg'
f = Image.open(path)
print(f.size)
print(f.mode)
print(f.format)
# f.show()

T.RandomHorizontalFlip()