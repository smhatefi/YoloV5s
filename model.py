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

class YOLOv5s(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv5s, self).__init__()
        self.num_classes = num_classes

        # YOLOv5s Backbone
        self.stem = Conv(3, 32, kernel_size=6, stride=2, padding=2)
        self.conv2 = Conv(32, 64, kernel_size=3, stride=2, padding=1)
        self.csp1 = CSPBottleneck(64, 64, n=1)
        self.conv3 = Conv(64, 128, kernel_size=3, stride=2, padding=1)
        self.csp2 = CSPBottleneck(128, 128, n=3)
        self.conv4 = Conv(128, 256, kernel_size=3, stride=2, padding=1)
        self.csp3 = CSPBottleneck(256, 256, n=3)
        self.conv5 = Conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.csp4 = CSPBottleneck(512, 512, n=1)
        self.spp = SPP(512, 512)

        # YOLOv5s Head
        self.head1 = Conv(512, 256, kernel_size=1, stride=1, padding=0)
        self.head2 = Conv(256, 128, kernel_size=1, stride=1, padding=0)
        self.head3 = Conv(128, 128, kernel_size=3, stride=2, padding=1)
        self.head4 = Conv(256, 256, kernel_size=3, stride=2, padding=1)
        self.head5 = Conv(512, 512, kernel_size=3, stride=2, padding=1)
        
        # Output layers for detection
        self.detect1 = nn.Conv2d(128, self.num_classes + 5, kernel_size=1)
        self.detect2 = nn.Conv2d(256, self.num_classes + 5, kernel_size=1)
        self.detect3 = nn.Conv2d(512, self.num_classes + 5, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv2(x)
        x = self.csp1(x)
        x = self.conv3(x)
        x = self.csp2(x)
        x = self.conv4(x)
        x = self.csp3(x)
        x = self.conv5(x)
        x = self.csp4(x)
        x = self.spp(x)
        
        x1 = self.head1(x)
        x2 = self.head2(x1)
        x3 = self.head3(x2)
        x4 = self.head4(x3)
        x5 = self.head5(x4)

        y1 = self.detect1(x3)
        y2 = self.detect2(x4)
        y3 = self.detect3(x5)

        return y1, y2, y3

# Instantiate the model
model = YOLOv5s(num_classes=80)
