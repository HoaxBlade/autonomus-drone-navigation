import torch
import torch.nn as nn
import torchvision.models as models
from .netvlad import NetVLAD

class VisualEncoder(nn.Module):
    def __init__(self, architecture='resnet50', use_netvlad=True):
        super(VisualEncoder, self).__init__()
        
        # Base architecture (ResNet as a feature extractor)
        if architecture == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            # Use layers up to the last convolutional block
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            in_dim = 2048
        elif architecture == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            in_dim = 512
        else:
            raise ValueError(f"Unrecognized architecture {architecture}")
            
        self.use_netvlad = use_netvlad
        if use_netvlad:
            self.netvlad = NetVLAD(num_clusters=64, dim=in_dim, alpha=100.0)
            self.output_dim = 64 * in_dim
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = in_dim

    def forward(self, x):
        x = self.features(x)
        if self.use_netvlad:
            x = self.netvlad(x)
        else:
            x = self.pool(x).view(x.size(0), -1)
        return x

class GoalEncoder(nn.Module):
    """
    Encoder for the "Target Goal Image". Often shares weights with the VisualEncoder.
    """
    def __init__(self, visual_encoder):
        super(GoalEncoder, self).__init__()
        self.encoder = visual_encoder

    def forward(self, x):
        return self.encoder(x)
