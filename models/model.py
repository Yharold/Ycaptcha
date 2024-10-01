from turtle import st
from torch import nn
import torch
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class RawResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=62,
        norm_layer=None,
    ):
        super(RawResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)    

    def make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                norm_layer,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4


def ResNet18(**kwargs):
    return RawResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


class ResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        dropout=0.5,
        num_classes=62,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)    

    def make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                norm_layer,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4
