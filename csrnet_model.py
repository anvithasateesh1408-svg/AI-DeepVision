import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_vgg_weights=True):
        super(CSRNet, self).__init__()

        # =====================================================
        # FRONTEND (VGG16 first 23 layers)
        # =====================================================
        if load_vgg_weights:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = models.vgg16(weights=None)

        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:23]
        )

        # =====================================================
        # BACKEND (Dilated Convolutions – CSRNet standard)
        # =====================================================
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Initialize ONLY when training from scratch
        if not load_vgg_weights:
            self._initialize_weights()

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    # =====================================================
    # WEIGHT INIT (training only)
    # =====================================================
    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
