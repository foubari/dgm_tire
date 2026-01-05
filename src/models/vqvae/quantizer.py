"""
Vector Quantizer with EMA updates for VQ-VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA updates.
    
    Implements straight-through estimator for backpropagation.
    """
    
    def __init__(self, num_embeddings=512, embedding_dim=20, commitment_cost=0.25, 
                 ema_decay=0.99, epsilon=1e-5):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize embeddings uniformly
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
        # EMA tracking for codebook updates
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embedding.weight.data.clone())
        
        # Track usage for codebook reset
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.register_buffer('last_reset_usage', torch.zeros(num_embeddings))  # Track usage at last reset
    
    def forward(self, z_e):
        """
        Quantize continuous latent z_e to discrete codes.
        
        Args:
            z_e: (B, embedding_dim, H, W) - continuous encoder output
        
        Returns:
            z_q: (B, embedding_dim, H, W) - quantized latent (for decoder)
            indices: (B, H, W) - discrete code indices
            losses: tuple of (vq_loss, commitment_loss)
        """
        # Flatten spatial dimensions
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        z_e_flat = z_e_flat.view(-1, D)  # (B*H*W, D)
        
        # Compute distances to codebook
        distances = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        )  # (B*H*W, num_embeddings)
        
        # Find closest codebook entries
        indices_flat = torch.argmin(distances, dim=1)  # (B*H*W,)
        indices = indices_flat.view(B, H, W)  # (B, H, W)
        
        # Quantize
        z_q_flat = self.embedding(indices_flat)  # (B*H*W, D)
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        
        # Compute losses
        # VQ loss: ||sg[z_e] - z_q||^2
        vq_loss = F.mse_loss(z_q, z_e.detach())
        
        # Commitment loss: β||z_e - sg[z_q]||^2
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        
        # EMA update (only during training)
        if self.training:
            self._ema_update(z_e_flat, indices_flat)
        
        return z_q_st, indices, (vq_loss, commitment_loss)
    
    def _ema_update(self, z_e_flat, indices_flat):
        """Update codebook using EMA."""
        # Count usage of each code
        encodings = F.one_hot(indices_flat, self.num_embeddings).float()  # (B*H*W, num_embeddings)
        cluster_size = encodings.sum(0)  # (num_embeddings,)
        
        # Update EMA cluster size
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        
        # Compute sum of embeddings per cluster
        embedding_sum = torch.matmul(encodings.t(), z_e_flat)  # (num_embeddings, D)
        
        # Update EMA embedding average
        self.ema_embed_avg.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)
        
        # Update codebook weights
        # FIX: Use cluster size in counts (with Laplace smoothing), not normalized distribution
        # ema_embed_avg is a SUM, so we need to divide by cluster SIZE (counts), not normalized distribution
        cluster_size = self.ema_cluster_size + self.epsilon
        self.embedding.weight.data.copy_(
            self.ema_embed_avg / cluster_size.unsqueeze(1)
        )
        
        # Track usage for reset
        self.usage_count.add_(cluster_size)
    
    def reset_unused_codes(self):
        """Reset codes that haven't been used since last reset."""
        # Find codes that haven't been used since last reset
        # (codes with usage_count == last_reset_usage)
        unused_mask = self.usage_count == self.last_reset_usage

        if unused_mask.sum() > 0:
            # Reinitialize unused codes randomly near used codes
            # Use small random values to avoid disrupting training
            num_unused = unused_mask.sum().item()
            self.embedding.weight.data[unused_mask] = torch.randn(
                num_unused, self.embedding_dim, device=self.embedding.weight.device
            ) * 0.01

        # Update last reset usage tracking (always, even if no codes were reset)
        self.last_reset_usage.copy_(self.usage_count)
    
    def get_codebook_usage(self):
        """Get statistics about codebook usage."""
        used_codes = (self.usage_count > 0).sum().item()
        usage_rate = used_codes / self.num_embeddings
        return {
            'used_codes': used_codes,
            'total_codes': self.num_embeddings,
            'usage_rate': usage_rate
        }

