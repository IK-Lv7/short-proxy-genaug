import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """ResNet for 32x32 inputs (3x3 first conv, no maxpool)."""

    def __init__(self, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


class VGGCIFAR(nn.Module):
    """VGG for 32x32 / 64x64 inputs (with BatchNorm, no residual connections)."""

    CFG = {"vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]}

    def __init__(self, name, num_classes=100):
        super().__init__()
        layers, c = [], 3
        for v in self.CFG[name]:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def features(self, x):
        return F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


class PretrainedResNet50(nn.Module):
    """torchvision ImageNet-pretrained ResNet-50 (IMAGENET1K_V2). Replace only the last layer and fine-tune everything."""

    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        self.fc = net.fc

    def features(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return n.avgpool(x).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


class PretrainedViTB16(nn.Module):
    """torchvision ImageNet-pretrained ViT-B/16 (IMAGENET1K_V1). Replace only the classification head and fine-tune everything."""

    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        net.heads = nn.Identity()
        self.net = net
        self.fc = nn.Linear(net.hidden_dim, num_classes)

    def features(self, x):
        return self.net(x)  # heads removed, so this returns the CLS token feature

    def forward(self, x):
        return self.fc(self.features(x))


def build_model(cfg, num_classes=None):
    if num_classes is None:
        from .data import dataset_info
        num_classes = dataset_info(cfg["data"])["num_classes"]
    name = cfg["model"]["name"]
    if name == "resnet18_cifar":
        return ResNetCIFAR([2, 2, 2, 2], num_classes)
    if name == "vitb16_pretrained":
        return PretrainedViTB16(num_classes)
    if name == "resnet50_pretrained":
        return PretrainedResNet50(num_classes)
    if name == "vgg11_cifar":
        return VGGCIFAR("vgg11", num_classes)
    raise ValueError(f"unknown model: {name}")
