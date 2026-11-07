"""OWL-GPS relevance encoder and distributional concept weighting.

The RelevanceEncoder2 architecture is preserved from the research code for
checkpoint compatibility. The lighter PooledRelevanceEncoder supports demos.
"""
import torch
from torch import nn
from torch.nn import functional as F

class RelevanceEncoder2(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        """
        Relevance Encoder that predicts:
        - mu: Mean relevance vector (B, 33), which assigns importance to each of the 45 encoders.
        - sigma: Standard deviation vector (B, 33), which represents uncertainty for each encoder.

        Arguments:
        - embed_dim: Number of channels in each input embedding (e.g., 768).
        - hidden_dim: Number of hidden channels for convolutional layers.
        """
        super(RelevanceEncoder2, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1)

        
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.mu_layer = nn.Linear(hidden_dim * 64 * 64, 1)  
        self.sigma_layer = nn.Linear(hidden_dim * 64 * 64, 1)  

    def forward(self, x):
        """
        Forward pass of the Relevance Encoder.

        Arguments:
        - x: Tensor of shape (B, 33, embed_dim, 256, 256), where:
          - B: Batch size
          - 33: Number of different encoders
          - embed_dim: Feature embedding size (e.g., 768)
          - 256x256: Spatial dimensions

        Returns:
        - mu: Mean relevance vector, shape (B, 33).
        - sigma: Standard deviation vector, shape (B, 33).
        """
        B, N, C, H, W = x.shape  

        
        mu_list = []
        sigma_list = []

        for i in range(N):
            encoder_output = x[:, i, :, :, :]  

            
            encoder_output = F.relu(self.bn1(self.conv1(encoder_output)))
            encoder_output = F.relu(self.bn2(self.conv2(encoder_output)))
            encoder_output = F.relu(self.bn3(self.conv3(encoder_output)))

            
            encoder_output = self.pool(encoder_output)  
            encoder_output = self.pool(encoder_output)  

            
            encoder_output = torch.flatten(encoder_output, start_dim=1)  

            
            mu = F.softplus(self.mu_layer(encoder_output))  
            sigma = F.softplus(self.sigma_layer(encoder_output)) + 1e-6  

            mu_list.append(mu)
            sigma_list.append(sigma)

        
        mu = torch.cat(mu_list, dim=1)  
        sigma = torch.cat(sigma_list, dim=1)  


        return mu, sigma

class PooledRelevanceEncoder(nn.Module):
    """Memory-light relevance CVAE head for examples; NOT checkpoint compatible."""
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(hidden_dim, 1)
        self.sigma_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        
        pooled = features.mean(dim=(-2, -1))
        z = self.projection(pooled)
        return F.softplus(self.mu_layer(z)).squeeze(-1), (
            F.softplus(self.sigma_layer(z)).squeeze(-1) + 1e-6
        )

def sample_relevance(mu: torch.Tensor, sigma: torch.Tensor, *, stochastic: bool = True):
    """Sample the CVAE relevance vector and normalize across concept channels."""
    eps = torch.randn_like(sigma) if stochastic else torch.zeros_like(sigma)
    return torch.softmax(mu + eps * sigma, dim=-1)

def orthogonalize_concepts(features: torch.Tensor, eps: float = 1e-6):
    """Pixelwise Gram–Schmidt along the concept axis.

    features: B x K x C x H x W. The C-dimensional vectors are made
    orthogonal at each spatial location, mirroring gram_schmidt_optimized.
    """
    if features.ndim != 5:
        raise ValueError("features must have shape (batch, concepts, channels, H, W)")
    vecs = features.float().unbind(dim=1)
    orthogonal = []
    for vec in vecs:
        residual = vec
        for basis in orthogonal:
            dot = (residual * basis).sum(dim=1, keepdim=True)
            norm = basis.square().sum(dim=1, keepdim=True).clamp_min(eps)
            residual = residual - (dot / norm) * basis
        orthogonal.append(residual)
    return torch.stack(orthogonal, dim=1).to(dtype=features.dtype)
