import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD(nn.Module):
    """
    NetVLAD layer implementation in PyTorch.
    Reference: Arandjelovic et al., "NetVLAD: CNN architecture for weakly supervised place recognition"
    """
    def __init__(self, num_clusters=64, dim=128, alpha=100.0, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of soft-assignment. A higher value leads to hard-assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor-dim

        # Soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # Vectorized residual calculation: (N, K, C, L)
        # x_flatten: (N, C, L), centroids: (K, C)
        # residual[n, k, c, l] = x_flatten[n, c, l] - centroids[k, c]
        residual = x_flatten.unsqueeze(1) - self.centroids.view(1, self.num_clusters, C, 1)
        
        # Apply soft assignment weights and sum over spatial dimensions (L)
        # soft_assign.unsqueeze(2): (N, K, 1, L)
        vlad = torch.sum(residual * soft_assign.unsqueeze(2), dim=-1) # (N, K, C)

        vlad = F.normalize(vlad, p=2, dim=2)  # Intra-normalization
        vlad = vlad.view(x.size(0), -1)       # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
