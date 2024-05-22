import torch
import torch.nn as nn
from .common import Conv, CSPBottleneck, SPP

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

# Utility function to load official weights
def load_official_weights(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_weights = checkpoint['model'].state_dict()

    new_state_dict = {}
    for k, v in model_weights.items():
        name = k
        if name.startswith('model.'):
            name = name[6:]  # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model
