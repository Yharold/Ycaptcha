import os, re, sys

sys.path.append("E:\Code\Ycaptcha")
from config.parameter import *
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


def str2label(label_str):
    label = []
    for i in range(0, char_number):
        if label_str[i] >= "0" and label_str[i] <= "9":
            label.append(ord(label_str[i]) - ord("0"))
        elif label_str[i] >= "a" and label_str[i] <= "z":
            label.append(ord(label_str[i]) - ord("a") + 10)
        else:
            label.append(ord(label_str[i]) - ord("A") + 36)
    return label


def label2str(label):
    label_str = ""
    for c in label:
        if c < 10:
            label_str += chr(c + ord("0"))
        elif c < 36:
            label_str += chr(c - 10 + ord("a"))
        else:
            label_str += chr(c - 36 + ord("A"))
    return label_str


class AugedCaptcha(data.Dataset):
    def __init__(self, root):
        self.image_path = [os.path.join(root, img) for img in os.listdir(root)]
        self.trans = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        img_path = self.image_path[index]
        img_name = os.path.basename(img_path)
        pattern = re.compile(r"train_original_(\d*\w*)")
        label = pattern.search(img_name).groups()[0]
        label = torch.Tensor(str2label(label))
        data = Image.open(img_path)
        data = self.trans(data)
        return data, label

    def __len__(self):
        return len(self.image_path)


class Captcha(data.Dataset):
    def __init__(self, root):
        self.image_path = [os.path.join(root, img) for img in os.listdir(root)]
        self.trans = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        img_path = self.image_path[index]
        img_name = os.path.basename(img_path)
        label = img_name.split(".")[0]
        label = torch.Tensor(str2label(label))
        data = Image.open(img_path)
        data = self.trans(data)
        return data, label

    def __len__(self):
        return len(self.image_path)


if __name__ == "__main__":
    # image_path = r"E:\Code\Ycaptcha\datasets\auged_train_0"
    # dataset = AugedCaptcha(image_path)
    # image_path = r"E:\Code\Ycaptcha\datasets\train"
    image_path = r"E:\Code\Ycaptcha\datasets\test"
    dataset = Captcha(image_path)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, (data, label) in enumerate(dataloader):
        print(i, data.shape, label.shape)
        if i == 10:
            break
