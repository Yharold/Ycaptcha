import os
from lib.dataset import label2str
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from config.parameter import *


# 输入一个验证码图片路径和模型，预测标签
# 返回验证码图片，真实标签，预测标签
def predict(model, image_path):
    model.eval()
    real_label = os.path.basename(image_path).split(".")[0]
    if not os.path.exists(image_path):
        raise ValueError("Image path does not exist")
    image = Image.open(image_path)
    trans = transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    data = trans(image)
    y1, y2, y3, y4 = model(data.unsqueeze(0))
    y1, y2, y3, y4 = (
        y1.topk(1, dim=1)[-1].view(1, 1),
        y2.topk(1, dim=1)[-1].view(1, 1),
        y3.topk(1, dim=1)[-1].view(1, 1),
        y4.topk(1, dim=1)[-1].view(1, 1),
    )
    y = torch.cat((y1, y2, y3, y4), dim=1)
    dec_label = label2str([y[0][0], y[0][1], y[0][2], y[0][3]])
    return image, real_label, dec_label


if __name__ == "__main__":
    from models.model import ResNet
    import random

    image_dir = r"E:\Code\Ycaptcha\datasets\test"
    image_path_list = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")
    ]
    n = 16
    image_paths = random.sample(image_path_list, n)

    model = ResNet()
    weight_path = r"weights\20240927-165911\112_ResNet_AdamW_0.001_200_0.pth"
    model.load_state_dict(torch.load(weight_path))
    data = []
    for img_path in image_paths:
        image, real_label, dec_label = predict(model, img_path)
        data.append((image, real_label, dec_label))
    r = n // 4 if n % 4 == 0 else n // 4 + 1
    c = 4
    fig, axes = plt.subplots(r, c)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(data):
            ax.imshow(data[i][0], cmap="viridis")
            ax.set_title(f"{data[i][1]}->{data[i][2]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
