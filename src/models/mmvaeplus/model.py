"""
Base MMVAEplus class definition.
"""

import torch
import torch.nn as nn

from .utils import get_mean


class MMVAEplus(nn.Module):
    """
    MMVAEplus class definition.
    """
    
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAEplus, self).__init__()
        self.pz = prior_dist  # Prior distribution
        self.pw = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])  # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass
        self.params = params  # Model parameters (i.e. args passed to main script)

    @staticmethod
    def getDataSets(batch_size, shuffle=True, device="cuda"):
        # Handle getting individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, cond=None, K=1):
        """
        Forward function with optional conditioning.
        
        Input:
            - x: list of data samples for each modality
            - cond: list of condition tensors (one per modality) or None
            - K: number of samples for reparameterization in latent space
        
        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of self- and cross- reconstructions. px_us[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note these latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            cond_m = cond[m] if cond is not None else None
            qu_x, px_u, us = vae(x[m], cond=cond_m, K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality d
                    pw = vae.pw(*vae.pw_params_aux)
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                    # Fixed for cuda
                    if not getattr(self.params, 'no_cuda', False) and torch.cuda.is_available():
                        latents_w = latents_w.cuda()
                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=-1)
                    # Get cross-reconstruction likelihood (with conditioning)
                    cond_d = cond[d] if cond is not None else None
                    px_us[e][d] = vae.px_u(*vae.dec(us_combined, cond=cond_d))
        
        return qu_xs, px_us, uss

    def generate_unconditional(self, N, cond=None):
        """
        Unconditional generation with optional conditioning.
        
        Args:
            N: Number of samples to generate
            cond: Optional list of condition tensors (one per modality) or None
        
        Returns:
            Generations: list of tensors, one for each modality
        """
        with torch.no_grad():
            data = []
            # Sample N shared latents
            pz = self.pz(*self.pz_params)
            latents_z = pz.rsample(torch.Size([N]))  # (N, 1, latent_dim_z) due to batch_shape
            # Squeeze middle dimension (batch_shape causes extra dim)
            if latents_z.dim() == 3:
                latents_z = latents_z.squeeze(1)  # (N, latent_dim_z)
            
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z.size()[0]])  # (N, 1, latent_dim_w) due to batch_shape
                # Squeeze middle dimension
                if latents_w.dim() == 3:
                    latents_w = latents_w.squeeze(1)  # (N, latent_dim_w)
                
                latents = torch.cat((latents_w, latents_z), dim=-1)  # (N, latent_dim_w + latent_dim_z)
                cond_d = cond[d] if cond is not None else None
                
                # Verify cond_d has correct shape if provided
                if cond_d is not None:
                    # cond_d should be (N, cond_dim)
                    if cond_d.dim() != 2:
                        raise ValueError(
                            f"Conditioning should be 2D tensor, got {cond_d.dim()}D: {cond_d.shape}"
                        )
                    if cond_d.shape[0] != N:
                        raise ValueError(
                            f"Conditioning batch size mismatch: expected N={N}, got {cond_d.shape[0]}"
                        )
                    # Check that cond_d is not accidentally latents (which would have shape (N, latent_dim_u))
                    # Get cond_dim from decoder
                    expected_cond_dim = vae.dec.cond_dim if hasattr(vae.dec, 'cond_dim') else None
                    if expected_cond_dim is not None and cond_d.shape[1] != expected_cond_dim:
                        raise ValueError(
                            f"Conditioning feature dimension mismatch: expected cond_dim={expected_cond_dim}, "
                            f"got {cond_d.shape[1]}. Full shape: {cond_d.shape}. "
                            f"This might indicate cond_d is actually latents (shape should be (N, {expected_cond_dim}))"
                        )
                
                px_u = vae.px_u(*vae.dec(latents, cond=cond_d))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def self_and_cross_modal_generation_forward(self, data, cond=None, K=1):
        """
        Test-time self- and cross-model generation forward function with conditioning.
        
        Args:
            data: List of input data for each modality
            cond: List of condition tensors (one per modality) or None
            K: Number of samples
        
        Returns:
            Unimodal encoding distribution, Matrix of self- and cross-modal reconstruction distributions, Latent embeddings
        """
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        
        for m, vae in enumerate(self.vaes):
            cond_m = cond[m] if cond is not None else None
            qu_x, px_u, us = vae(data[m], cond=cond_m, K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        
        for e, us in enumerate(uss):
            latents_w, latents_z = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            for d, vae in enumerate(self.vaes):
                mean_w, scale_w = self.pw_params
                pw = self.pw(mean_w, scale_w)
                latents_w_new = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                us_new = torch.cat((latents_w_new, latents_z), dim=-1)
                if e != d:  # fill-in off-diagonal
                    cond_d = cond[d] if cond is not None else None
                    px_us[e][d] = vae.px_u(*vae.dec(us_new, cond=cond_d))
        
        return qu_xs, px_us, uss

    def self_and_cross_modal_generation(self, data, cond=None):
        """
        Test-time self- and cross-reconstruction with conditioning.
        
        Args:
            data: List of input data for each modality
            cond: List of condition tensors (one per modality) or None
        
        Returns:
            Matrix of self- and cross-modal reconstructions
        """
        with torch.no_grad():
            _, px_us, _ = self.self_and_cross_modal_generation_forward(data, cond=cond)
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_u) for px_u in r] for r in px_us]
        return recons

