"""
Meta VAE model for EpureDGM.

Hierarchical VAE architecture:
- Meta encoder: Encodes all components into single latent code
- Meta decoder: Generates component-specific latent codes
- Marginal decoders: Pre-trained frozen decoders for each component

Adapted from ICTAI for (64, 32) images and conditional generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MetaEncoder(nn.Module):
    """
    Meta-VAE encoder with conditioning support.

    Encodes all components into a single meta latent code.
    Uses shared backbone + component-specific branches.
    """
    def __init__(
        self,
        num_components: int = 5,
        latent_dim: int = 32,
        component_latent_dim: int = 4,
        cond_dim: int = 0
    ):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.component_latent_dim = component_latent_dim
        self.cond_dim = cond_dim

        # Shared backbone for each component (process 1-channel component)
        # Input: (1, 64, 32) → Output: flattened features
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=4, stride=2, padding=1),  # 64x32 → 32x16
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),  # 32x16 → 16x8
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),  # 16x8 → 8x4
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 128, kernel_size=4, stride=2, padding=1),  # 8x4 → 4x2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()  # 128 * 4 * 2 = 1024
        )

        # Component-specific encoders
        self.component_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 * 4 * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 192),
                nn.BatchNorm1d(192),
                nn.ReLU(),
                nn.Linear(192, 192)
            ) for _ in range(num_components)
        ])

        # Combine all component features
        combine_input_dim = 192 * num_components
        if cond_dim > 0:
            combine_input_dim += 256  # Add conditioning embedding size

            # Conditioning MLP
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
        else:
            self.cond_mlp = None

        self.fc_combine = nn.Sequential(
            nn.Linear(combine_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(384, latent_dim)
        self.fc_logvar = nn.Linear(384, latent_dim)

    def forward(
        self,
        components: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            components: [B, num_components, 64, 32] - all component images
            mask: [B, num_components] - binary mask (1=available, 0=masked)
            cond: [B, cond_dim] - conditioning vector

        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        B = components.size(0)
        features = []

        for i in range(self.num_components):
            # Extract single component: [B, 1, 64, 32]
            component = components[:, i:i+1]

            # Shared feature extraction
            shared_feat = self.shared_backbone(component)  # [B, 1024]

            # Component-specific processing
            feat = self.component_encoders[i](shared_feat)  # [B, 192]

            # Apply mask if provided
            if mask is not None:
                feat = feat * mask[:, i:i+1]

            features.append(feat)

        # Concatenate all component features
        combined = torch.cat(features, dim=1)  # [B, 192 * num_components]

        # Add conditioning if provided
        if self.cond_mlp is not None and cond is not None:
            cond_emb = self.cond_mlp(cond)  # [B, 256]
            combined = torch.cat([combined, cond_emb], dim=1)

        # Generate mean and log variance
        h = self.fc_combine(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class MetaDecoder(nn.Module):
    """
    Meta-VAE decoder with conditioning support.

    Generates component-specific latent codes from meta latent code.
    """
    def __init__(
        self,
        num_components: int = 5,
        latent_dim: int = 32,
        component_latent_dim: int = 4,
        cond_dim: int = 0
    ):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.component_latent_dim = component_latent_dim
        self.cond_dim = cond_dim

        # Initial projection
        initial_input_dim = latent_dim
        if cond_dim > 0:
            initial_input_dim += 256

            # Conditioning MLP
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
        else:
            self.cond_mlp = None

        self.fc_initial = nn.Linear(initial_input_dim, 512)

        # Parallel decoder blocks for each component's latent code
        self.component_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, component_latent_dim)
            ) for _ in range(num_components)
        ])

    def forward(
        self,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            z: Meta latent code [B, latent_dim]
            cond: [B, cond_dim] - conditioning vector

        Returns:
            List of component latent codes [B, component_latent_dim] each
        """
        # Add conditioning if provided
        if self.cond_mlp is not None:
            if cond is not None:
                cond_emb = self.cond_mlp(cond)
            else:
                # Use zero conditioning for unconditional generation
                cond_emb = torch.zeros(z.size(0), 256, device=z.device, dtype=z.dtype)
            z = torch.cat([z, cond_emb], dim=1)

        # Initial projection
        h = F.relu(self.fc_initial(z))

        # Generate latent code for each component
        component_latents = []
        for decoder in self.component_decoders:
            z_component = decoder(h)
            component_latents.append(z_component)

        return component_latents


class MarginalDecoder(nn.Module):
    """
    Simple decoder for individual components.

    Used during marginal training phase (Stage 1).
    """
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim

        # Projection to spatial features
        self.fc = nn.Linear(latent_dim, 128 * 4 * 2)

        # Transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),  # 4x2 → 8x4
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),  # 8x4 → 16x8
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),  # 16x8 → 32x16
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 1, kernel_size=4, stride=2, padding=1),  # 32x16 → 64x32
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
        Returns:
            [B, 1, 64, 32]
        """
        h = self.fc(z)
        h = h.view(-1, 128, 4, 2)
        return self.decoder(h)


class MetaVAE(nn.Module):
    """
    Complete Meta-VAE with conditioning support.

    Coordinates pre-trained marginal decoders via meta encoder/decoder.
    """
    def __init__(
        self,
        marginal_decoders: List[nn.Module],
        latent_dim: int = 32,
        component_latent_dim: int = 4,
        beta: float = 1.0,
        cond_dim: int = 0
    ):
        super().__init__()
        self.num_components = len(marginal_decoders)
        self.latent_dim = latent_dim
        self.component_latent_dim = component_latent_dim
        self.beta = beta
        self.cond_dim = cond_dim

        # Meta encoder and decoder
        self.meta_encoder = MetaEncoder(
            self.num_components,
            latent_dim,
            component_latent_dim,
            cond_dim
        )
        self.meta_decoder = MetaDecoder(
            self.num_components,
            latent_dim,
            component_latent_dim,
            cond_dim
        )

        # Pre-trained marginal decoders (frozen)
        self.marginal_decoders = nn.ModuleList(marginal_decoders)
        for decoder in self.marginal_decoders:
            for param in decoder.parameters():
                param.requires_grad = False

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        components: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ):
        """
        Args:
            components: [B, num_components, 64, 32]
            mask: [B, num_components] - binary mask
            cond: [B, cond_dim] - conditioning vector

        Returns:
            reconstructed: [B, num_components, 64, 32]
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
            z_meta: [B, latent_dim]
            z_components: List of [B, component_latent_dim]
        """
        # Encode to meta latent space
        mu, logvar = self.meta_encoder(components, mask, cond)
        z_meta = self.reparameterize(mu, logvar)

        # Generate component latent codes
        z_components = self.meta_decoder(z_meta, cond)

        # Generate components using marginal decoders
        reconstructed_list = []
        for z_comp, decoder in zip(z_components, self.marginal_decoders):
            recon = decoder(z_comp)  # [B, 1, 64, 32]
            reconstructed_list.append(recon)

        # Stack into single tensor
        reconstructed = torch.cat(reconstructed_list, dim=1)  # [B, num_components, 64, 32]

        return reconstructed, mu, logvar, z_meta, z_components

    def loss_function(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Meta-VAE loss with optional masking.

        Args:
            recon: [B, num_components, 64, 32]
            target: [B, num_components, 64, 32]
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
            mask: [B, num_components]

        Returns:
            total_loss: scalar
            loss_dict: dict with loss components
        """
        B = mu.size(0)

        # Component-wise reconstruction loss
        recon_loss = 0.0
        component_losses = []

        for i in range(self.num_components):
            # MSE for this component
            comp_loss = F.mse_loss(
                recon[:, i],
                target[:, i],
                reduction='none'
            )
            comp_loss = comp_loss.view(B, -1).mean(dim=1)  # [B]

            # Apply mask if provided
            if mask is not None:
                comp_loss = comp_loss * mask[:, i]

            component_losses.append(comp_loss.mean().item())
            recon_loss += comp_loss.sum()

        recon_loss = recon_loss / B

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
            'component_losses': component_losses
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate new complete systems.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            cond: [num_samples, cond_dim] - conditioning vectors

        Returns:
            [num_samples, num_components, 64, 32]
        """
        # Sample from prior
        z_meta = torch.randn(num_samples, self.latent_dim, device=device)

        # Generate component latent codes
        z_components = self.meta_decoder(z_meta, cond)

        # Generate components
        generated_list = []
        for z_comp, decoder in zip(z_components, self.marginal_decoders):
            component = decoder(z_comp)  # [num_samples, 1, 64, 32]
            generated_list.append(component)

        generated = torch.cat(generated_list, dim=1)  # [num_samples, num_components, 64, 32]
        return generated

    @torch.no_grad()
    def inpaint(
        self,
        partial_components: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate missing components (inpainting/cross-modal generation).

        Args:
            partial_components: [B, num_components, 64, 32] - with zeros for missing
            mask: [B, num_components] - 1=available, 0=missing
            cond: [B, cond_dim] - conditioning vectors
            num_samples: Number of samples per input (for stochastic generation)

        Returns:
            [B*num_samples, num_components, 64, 32]
        """
        B = partial_components.size(0)

        # Encode with mask
        mu, logvar = self.meta_encoder(partial_components, mask, cond)

        # Generate multiple samples if requested
        all_samples = []
        for _ in range(num_samples):
            z_meta = self.reparameterize(mu, logvar)

            # Expand conditioning if needed
            cond_sample = cond
            if cond is not None and num_samples > 1:
                cond_sample = cond  # Use same cond for each sample

            z_components = self.meta_decoder(z_meta, cond_sample)

            # Generate all components
            generated_list = []
            for z_comp, decoder in zip(z_components, self.marginal_decoders):
                component = decoder(z_comp)
                generated_list.append(component)

            generated = torch.cat(generated_list, dim=1)
            all_samples.append(generated)

        # Stack samples
        result = torch.cat(all_samples, dim=0)  # [B*num_samples, num_components, 64, 32]
        return result
