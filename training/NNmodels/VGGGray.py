import torch.nn as nn
import torchvision.models as models

# VGG16 Gray version
class VGG16Gray(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16Gray, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)

# VGG19 Gray version
class VGG19Gray(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19Gray, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg19.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg19(x)