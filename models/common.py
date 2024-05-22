import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activation=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class CSPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super(CSPBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(hidden_channels * (len(kernel_sizes) + 1), out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))
