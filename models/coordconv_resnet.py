import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, device="cpu"):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.device = device

    def forward(self, input_tensor):
        if self.rank == 2:
            b, c, y, x = input_tensor.shape
            x_ones = torch.ones([1, 1, 1, x], dtype=torch.int32)
            y_ones = torch.ones([1, 1, 1, y], dtype=torch.int32)

            x_range = torch.arange(y, dtype=torch.int32)
            y_range = torch.arange(x, dtype=torch.int32)
            x_range = x_range[None, None, :, None]
            y_range = y_range[None, None, :, None]

            x_channel = torch.matmul(x_range, x_ones)
            y_channel = torch.matmul(y_range, y_ones)

            # transpose y
            y_channel = y_channel.permute(0, 1, 3, 2)

            # normalization
            x_channel = x_channel.float() / (y - 1)
            y_channel = y_channel.float() / (x - 1)

            x_channel = x_channel * 2 - 1
            y_channel = y_channel * 2 - 1

            x_channel = x_channel.repeat(b, 1, 1, 1)
            y_channel = y_channel.repeat(b, 1, 1, 1)

            # put tensor to the device
            x_channel = x_channel.to(self.device)
            y_channel = y_channel.to(self.device)

            out = torch.cat([input_tensor, x_channel, y_channel], dim=1)
            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(x_channel - 0.5, 2) +
                    torch.pow(y_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 with_r=False,
                 device="cpu"):
        super(CoordConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, device)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r),
                              out_channels, kernel_size, stride, padding,
                              dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CoordResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, device="cpu"):
        super(CoordResNet, self).__init__()
        self.in_planes = 64

        self.coordconv = CoordConv2d(
            3, 8, 1, bias=False, with_r=True, device=device)
        self.conv1 = nn.Conv2d(
            8, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.coordconv(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def CoordResNet18(device="cpu"):
    return CoordResNet(BasicBlock, [2, 2, 2, 2], device=device)


def CoordResNet34(device="cpu"):
    return CoordResNet(BasicBlock, [3, 4, 6, 3], device=device)


def CoordResNet50(device="cpu"):
    return CoordResNet(Bottleneck, [3, 4, 6, 3], device=device)


def CoordResNet101(device="cpu"):
    return CoordResNet(Bottleneck, [3, 4, 23, 3], device=device)


def CoordResNet152(device="cpu"):
    return CoordResNet(Bottleneck, [3, 8, 36, 3], device=device)


def test():
    net = CoordResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__':
    test()
