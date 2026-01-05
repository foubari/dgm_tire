"""
PixelCNN Prior for VQ-VAE latent space modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for causal modeling in PixelCNN.
    
    Type A: Masks center pixel and all pixels after (for first layer)
    Type B: Includes center pixel, masks only future pixels (for subsequent layers)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='B', **kwargs):
        # Ensure padding is set to preserve spatial dimensions
        if 'padding' not in kwargs:
            if isinstance(kernel_size, int):
                kwargs['padding'] = kernel_size // 2
            else:
                kwargs['padding'] = (kernel_size[0] // 2, kernel_size[1] // 2)
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        
        # Create mask
        self.register_buffer('mask', self._create_mask())
    
    def _create_mask(self):
        """Create causal mask."""
        mask = torch.ones_like(self.weight)
        if isinstance(self.kernel_size, int):
            h, w = self.kernel_size, self.kernel_size
        else:
            h, w = self.kernel_size
        center_h, center_w = h // 2, w // 2
        
        # Mask all pixels after center
        for i in range(h):
            for j in range(w):
                if i < center_h or (i == center_h and j < center_w):
                    # Keep (before center)
                    mask[:, :, i, j] = 1.0
                elif i == center_h and j == center_w:
                    # Center pixel
                    if self.mask_type == 'A':
                        mask[:, :, i, j] = 0.0  # Mask center in type A
                    else:
                        mask[:, :, i, j] = 1.0  # Keep center in type B
                else:
                    # After center
                    mask[:, :, i, j] = 0.0
        
        return mask
    
    def forward(self, x):
        """Apply masked convolution."""
        # Apply mask non-destructively
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MaskedResidualBlock(nn.Module):
    """Residual block with masked convolution and gated activation."""
    
    def __init__(self, in_channels, out_channels, mask_type='B', kernel_size=3):
        super().__init__()
        
        # Gated activation: split channels, apply tanh and sigmoid
        self.gate_conv = MaskedConv2d(in_channels, out_channels * 2, kernel_size, mask_type=mask_type)
        
        # After gate, we have out_channels channels
        self.conv1 = MaskedConv2d(out_channels, out_channels, kernel_size, mask_type='B')
        self.conv2 = MaskedConv2d(out_channels, out_channels, kernel_size, mask_type='B')
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        """
        Forward with gated activation.
        
        Gated activation: split channels into two parts, apply tanh and sigmoid, multiply.
        """
        # Gated activation path
        gate = self.gate_conv(x)
        gate_tanh, gate_sigmoid = gate.chunk(2, dim=1)
        gate_out = torch.tanh(gate_tanh) * torch.sigmoid(gate_sigmoid)  # (B, out_channels, H, W)
        
        # Residual path
        residual = self.skip(x)
        
        # Main path
        out = self.conv1(gate_out)
        out = F.relu(out)
        out = self.conv2(out)
        
        return out + residual


class PixelCNNPrior(nn.Module):
    """
    PixelCNN Prior for modeling discrete latent distribution.
    
    Models p(z|cond) autoregressively using masked convolutions.
    """
    
    def __init__(
        self,
        num_embeddings=512,
        hidden_dim=128,
        cond_dim=2,
        num_layers=12,
        kernel_size=3,
        cond_drop_prob=0.1,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.cond_drop_prob = cond_drop_prob
        
        # Embedding layer for discrete codes
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        
        # Conditioning MLP
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.GELU(),
                nn.Linear(256, 256)
            )
            # Input to first layer: hidden_dim + 256 (conditioning)
            first_layer_in = hidden_dim + 256
        else:
            self.cond_mlp = None
            first_layer_in = hidden_dim
        
        # Stack of masked residual blocks
        self.layers = nn.ModuleList()
        
        # First layer uses type A mask
        self.layers.append(
            MaskedResidualBlock(first_layer_in, hidden_dim, mask_type='A', kernel_size=kernel_size)
        )
        
        # Subsequent layers use type B mask
        for _ in range(num_layers - 1):
            self.layers.append(
                MaskedResidualBlock(hidden_dim, hidden_dim, mask_type='B', kernel_size=kernel_size)
            )
        
        # Output layer: predict logits for each codebook entry
        self.out_conv = nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1)
    
    def forward(self, indices, cond=None):
        """
        Forward pass for training.
        
        Args:
            indices: (B, H, W) - discrete latent codes
            cond: (B, cond_dim) or None - conditioning vector
        
        Returns:
            logits: (B, num_embeddings, H, W) - predicted logits for each position
        """
        B, H, W = indices.shape
        
        # Embed discrete codes
        x = self.embedding(indices)  # (B, H, W, hidden_dim)
        x = x.permute(0, 3, 1, 2)  # (B, hidden_dim, H, W)
        
        # Apply conditioning (always concatenate if cond_mlp exists, even if cond is None)
        if self.cond_mlp is not None:
            if cond is not None:
                if self.training and self.cond_drop_prob > 0:
                    # Randomly drop conditioning during training
                    drop_mask = torch.rand(B, device=cond.device) > self.cond_drop_prob
                    cond = cond * drop_mask.unsqueeze(1).float()
                cond_emb = self.cond_mlp(cond)  # (B, 256)
            else:
                # Use zeros if cond is None (unconditional generation)
                cond_emb = torch.zeros(B, 256, device=x.device, dtype=x.dtype)
            
            cond_spatial = cond_emb[:, :, None, None].expand(-1, -1, H, W)  # (B, 256, H, W)
            x = torch.cat([x, cond_spatial], dim=1)  # (B, hidden_dim+256, H, W)
        
        # Apply masked convolutions
        for layer in self.layers:
            x = layer(x)
        
        # Output logits
        logits = self.out_conv(x)  # (B, num_embeddings, H, W)
        
        return logits
    
    @torch.no_grad()
    def sample(self, batch_size, cond=None, temperature=1.0, top_k=None, device='cuda', H=16, W=8):
        """
        Autoregressive sampling from the prior.
        
        Args:
            batch_size: Number of samples to generate
            cond: (B, cond_dim) or None - conditioning vector
            temperature: Sampling temperature (higher = more random)
            top_k: If not None, only sample from top-k logits
            device: Device to generate on
            H, W: Spatial dimensions of latent space
        
        Returns:
            indices: (B, H, W) - sampled discrete codes
        """
        self.eval()
        
        indices = torch.zeros((batch_size, H, W), dtype=torch.long, device=device)
        
        # Expand cond if provided
        if cond is not None:
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(batch_size, -1)
        else:
            cond = None
        
        # Generate pixel by pixel in raster scan order
        for h in range(H):
            for w in range(W):
                # Get logits for current position
                logits = self.forward(indices, cond)  # (B, num_embeddings, H, W)
                logits_hw = logits[:, :, h, w] / temperature  # (B, num_embeddings)
                
                # Optional top-k filtering
                if top_k is not None and top_k < logits_hw.shape[1]:
                    top_k_logits, top_k_indices = torch.topk(logits_hw, top_k, dim=1)
                    logits_hw = torch.full_like(logits_hw, float('-inf'))
                    logits_hw.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample from distribution
                probs = F.softmax(logits_hw, dim=-1)
                indices[:, h, w] = torch.multinomial(probs, 1).squeeze(-1)
        
        return indices
    
    @torch.no_grad()
    def inpaint(self, known_indices, mask, cond=None, temperature=1.0, top_k=None):
        """
        Inpaint missing regions in latent space.
        
        Args:
            known_indices: (B, H, W) - known indices (will be kept where mask=1)
            mask: (B, H, W) - 1 for known positions, 0 for unknown
            cond: (B, cond_dim) or None - conditioning vector
            temperature: Sampling temperature
            top_k: Top-k filtering
        
        Returns:
            indices: (B, H, W) - inpainted indices
        """
        self.eval()
        
        indices = known_indices.clone()
        B, H, W = indices.shape
        
        # Expand cond if provided
        if cond is not None:
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(B, -1)
        else:
            cond = None
        
        # Generate pixel by pixel, but skip known positions
        for h in range(H):
            for w in range(W):
                # Check if this position needs generation
                needs_gen = (mask[:, h, w] == 0).any()
                
                if needs_gen:
                    # Get logits for current position
                    logits = self.forward(indices, cond)  # (B, num_embeddings, H, W)
                    logits_hw = logits[:, :, h, w] / temperature  # (B, num_embeddings)
                    
                    # Optional top-k filtering
                    if top_k is not None and top_k < logits_hw.shape[1]:
                        top_k_logits, top_k_indices = torch.topk(logits_hw, top_k, dim=1)
                        logits_hw = torch.full_like(logits_hw, float('-inf'))
                        logits_hw.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Sample
                    probs = F.softmax(logits_hw, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    
                    # Only update positions that need generation
                    update_mask = mask[:, h, w] == 0
                    indices[update_mask, h, w] = sampled[update_mask]
        
        return indices
    
    def cfg_sample(self, batch_size, cond, guidance_scale=2.0, temperature=1.0, top_k=None, device='cuda', H=16, W=8):
        """
        Sample with classifier-free guidance.

        Optimized version that batches conditional and unconditional forward passes.

        Args:
            batch_size: Number of samples
            cond: (B, cond_dim) - conditioning vector
            guidance_scale: Guidance scale (1.0 = no guidance)
            temperature: Sampling temperature
            top_k: Top-k filtering
            device: Device
            H, W: Spatial dimensions

        Returns:
            indices: (B, H, W) - sampled codes
        """
        self.eval()

        indices = torch.zeros((batch_size, H, W), dtype=torch.long, device=device)

        # Expand cond
        if cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(batch_size, -1)

        # Generate pixel by pixel
        for h in range(H):
            for w in range(W):
                # Batch conditional and unconditional in a single forward pass
                # Stack indices: [conditional_batch, unconditional_batch]
                indices_batched = torch.cat([indices, indices], dim=0)

                # Stack conditions: [cond, None (zeros)]
                cond_batched = torch.cat([
                    cond,
                    torch.zeros_like(cond)
                ], dim=0)

                # Single forward pass
                logits_batched = self.forward(indices_batched, cond_batched)
                logits_batched_hw = logits_batched[:, :, h, w] / temperature

                # Split back into conditional and unconditional
                logits_cond_hw = logits_batched_hw[:batch_size]
                logits_uncond_hw = logits_batched_hw[batch_size:]

                # Classifier-free guidance: logits = logits_uncond + scale * (logits_cond - logits_uncond)
                logits_hw = logits_uncond_hw + guidance_scale * (logits_cond_hw - logits_uncond_hw)

                # Optional top-k filtering
                if top_k is not None and top_k < logits_hw.shape[1]:
                    top_k_logits, top_k_indices = torch.topk(logits_hw, top_k, dim=1)
                    logits_hw = torch.full_like(logits_hw, float('-inf'))
                    logits_hw.scatter_(1, top_k_indices, top_k_logits)

                # Sample
                probs = F.softmax(logits_hw, dim=-1)
                indices[:, h, w] = torch.multinomial(probs, 1).squeeze(-1)

        return indices

