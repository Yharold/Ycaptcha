from lib.dataset import label2str
import torch
from config.parameter import *
from lib.dataset import Captcha


def test(model, dataloader):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    total_num = len(dataloader)
    right_num = 0
    bad_list = []
    for idx, (data, label) in enumerate(dataloader):
        real_label = label2str(map(int, label[0]))
        label = label.long()
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        y1, y2, y3, y4 = model(data)
        y1, y2, y3, y4 = (
            y1.topk(1, dim=1)[1].view(1, 1),
            y2.topk(1, dim=1)[1].view(1, 1),
            y3.topk(1, dim=1)[1].view(1, 1),
            y4.topk(1, dim=1)[1].view(1, 1),
        )
        y = torch.cat([y1, y2, y3, y4], dim=1)
        dec_label = label2str([y[0][0], y[0][1], y[0][2], y[0][3]])
        if real_label == dec_label:
            right_num += 1
        else:
            bad_list.append((real_label, dec_label))
    accuracy = float(right_num) / float(total_num)
    return bad_list, accuracy


if __name__ == "__main__":
    from models.model import ResNet, ResNet18
    from matplotlib import pyplot as plt
    from PIL import Image
    import os

    #
    # weight_path = r"weights\20240927-165911\112_ResNet_AdamW_0.001_200_0.pth"
    weight_path = r"E:\Code\Ycaptcha\weights\20240930-184806\best.pth"
    # model = ResNet()
    model = ResNet18()
    model.load_state_dict(torch.load(weight_path))

    test_dataset = Captcha(root=test_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    bad_list, accuracy = test(model, test_dataloader)
    print(f"accuracy:{accuracy}")
    r = len(bad_list) // 4 if len(bad_list) % 4 == 0 else len(bad_list) // 4 + 1
    c = 4
    fig, axes = plt.subplots(r, c)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(bad_list):
            image_path = os.path.join(test_path, bad_list[i][0] + ".jpg")
            image = Image.open(image_path)
            ax.imshow(image)
            ax.set_title(f"{bad_list[i][0]}->{bad_list[i][1]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
