import torch
import torch.nn as nn
import torchvision.models as models
from .netvlad import NetVLAD

class PerceptionBackbone(nn.Module):
    """
    Unified ResNet backbone shared across all perception heads.
    """
    def __init__(self, architecture='resnet50'):
        super(PerceptionBackbone, self).__init__()
        if architecture == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.out_channels = 2048
        elif architecture == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x):
        return self.features(x)

class VisualEncoder(nn.Module):
    def __init__(self, backbone, use_netvlad=True):
        super(VisualEncoder, self).__init__()
        self.backbone = backbone
        self.use_netvlad = use_netvlad
        in_dim = backbone.out_channels
        
        if use_netvlad:
            self.netvlad = NetVLAD(num_clusters=64, dim=in_dim, alpha=100.0)
            self.output_dim = 64 * in_dim
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = in_dim

    def forward(self, x):
        x = self.backbone(x)
        if self.use_netvlad:
            x = self.netvlad(x)
        else:
            x = self.pool(x).view(x.size(0), -1)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, backbone):
        super(GoalEncoder, self).__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = backbone.out_channels

    def forward(self, x):
        x = self.backbone(x)
        return self.pool(x).view(x.size(0), -1)

class DepthEncoder(nn.Module):
    """
    Monocular Depth Estimation head branching off the shared backbone.
    Implement a simple decoder/upsampler for depth maps.
    """
    def __init__(self, backbone):
        super(DepthEncoder, self).__init__()
        self.backbone = backbone
        # Simple upsampling head to get back to HxW depth map
        self.decoder = nn.Sequential(
            nn.Conv2d(backbone.out_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Normalize depth to [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)
