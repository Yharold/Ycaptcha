from audioop import avg
import os
import time
import torch
from torch.utils.data import DataLoader, ConcatDataset
from lib.dataset import Captcha, AugedCaptcha
from config.parameter import *
from lib.optimizer import AdamW, RAdam
from torch.optim.lr_scheduler import StepLR
from lib.scheduler import GradualWarmupScheduler
from torchnet import meter
from torch.utils.tensorboard import SummaryWriter


auged_dataset = AugedCaptcha(auged_train_path)
train_dataset = Captcha(train_path)
test_dataset = Captcha(test_path)
merged_dataset = ConcatDataset([train_dataset, auged_dataset])
train_dataloader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(model):
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_after = StepLR(optimizer, step_size=20, gamma=0.5)
    scheduler = GradualWarmupScheduler(
        optimizer, 8, 10, after_scheduler=scheduler_after
    )
    loss_meter = meter.AverageValueMeter()
    avg_loss = []
    best_acc = -1.0

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_path, timestamp))
    writer.add_text("model", model.__class__.__name__)
    writer.add_text("optimizer", optimizer.__class__.__name__)
    writer.add_text("epoch", str(total_epoch))
    writer.add_text("batch_size", str(batch_size))
    writer.add_text("learning_rate", str(learning_rate))
    writer.add_text("weight_decay", str(weight_decay))
    writer.add_text("train_path", train_path)
    writer.add_text("test_path", test_path)
    writer.add_text("auged_train_path", auged_train_path)
    for epoch in range(total_epoch):
        model.train()
        loss_meter.reset()
        start_time = time.time()
        for idx, (data, label) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2, label3, label4 = (
                label[:, 0],
                label[:, 1],
                label[:, 2],
                label[:, 3],
            )
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(data)
            loss1, loss2, loss3, loss4 = (
                criterion(y1, label1),
                criterion(y2, label2),
                criterion(y3, label3),
                criterion(y4, label4),
            )
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if (idx + 1) % print_circle == 0:
                print(
                    "epoch:%04d|step:%03d|train loss:%.9f"
                    % (epoch, idx + 1, sum(avg_loss) / len(avg_loss))
                )
                avg_loss = []

        scheduler.step()
        accuracy, test_loss = test(model, test_dataloader)
        # 保存模型
        save_path = os.path.join(output_path, timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if best_acc < accuracy:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
        if (
            (epoch + 1) % save_circle == 0
            or epoch == total_epoch - 1
            or (best_acc > 0.8 and best_acc - accuracy < 0.001)
        ):
            model_name = str(epoch) + ".pth"
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
        # 记录日志

        writer.add_scalar("Loss/train_loss", loss_meter.value()[0], epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("test_accuracy", accuracy, epoch)
        spend_time = time.time() - start_time
        # 打印输出
        print("Learning rate: %.10f" % (scheduler.get_last_lr()[0]))
        print("test accuracy: %.5f" % accuracy)
        print("train loss:%.9f|test loss:%.9f" % (loss_meter.value()[0], test_loss))
        print(
            f"need time: {(total_epoch - epoch - 1) * spend_time / 60.0 :.2f} minutes"
        )


def test(model, dataloader):
    model.eval()
    total_num = len(os.listdir(test_path))
    right_num = 0
    criterion = torch.nn.CrossEntropyLoss()
    meter_loss = meter.AverageValueMeter()
    for idx, (data, label) in enumerate(dataloader):
        label = label.long()
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        y1, y2, y3, y4 = model(data)
        loss1, loss2, loss3, loss4 = (
            criterion(y1, label[:, 0]),
            criterion(y2, label[:, 1]),
            criterion(y3, label[:, 2]),
            criterion(y4, label[:, 3]),
        )
        loss = loss1 + loss2 + loss3 + loss4
        meter_loss.add(loss.item())
        bs = data.shape[0]
        y1, y2, y3, y4 = (
            y1.topk(1, dim=1)[1].view(bs, 1),
            y2.topk(1, dim=1)[1].view(bs, 1),
            y3.topk(1, dim=1)[1].view(bs, 1),
            y4.topk(1, dim=1)[1].view(bs, 1),
        )
        y = torch.cat([y1, y2, y3, y4], dim=1)
        # 4个字符中错误的设置为1，正确的设置为0，diff大小和y相等，(bs,4)
        diff = y != label
        # 统计错误的个数，正确的值为0，错误的大于0
        diff = diff.sum(dim=1)
        # 错误的为1，正确为0
        diff = diff != 0
        # 统计错误的个数
        res = diff.sum(0).item()
        # 统计正确的个数
        right_num += bs - res
    return float(right_num) / float(total_num), meter_loss.value()[0]


if __name__ == "__main__":
    from models.model import ResNet, ResNet18

    # models = [ResNet18(), ResNet()]
    # for model in models:
    #     train(model)
    model = ResNet()
    train(model)