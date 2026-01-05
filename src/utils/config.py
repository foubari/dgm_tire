"""
Configuration management for modular diffusion models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file (can be relative or absolute)
    
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    # Resolve relative paths
    if not config_path.is_absolute():
        # Try relative to current working directory
        if config_path.exists():
            config_path = config_path.resolve()
        else:
            # Try relative to src_new directory
            _THIS_FILE = Path(__file__).resolve()
            _SRC_NEW_DIR = _THIS_FILE.parent.parent
            potential_path = _SRC_NEW_DIR / config_path
            if potential_path.exists():
                config_path = potential_path.resolve()
            else:
                # Try configs subdirectory
                potential_path = _SRC_NEW_DIR / "configs" / config_path.name
                if potential_path.exists():
                    config_path = potential_path.resolve()
                else:
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_path}\n"
                        f"Searched in: {Path.cwd()}, {_SRC_NEW_DIR}, {_SRC_NEW_DIR / 'configs'}"
                    )
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    return config


def auto_complete_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-complete configuration by calculating missing values.
    
    Calculates:
    - model.channels from len(data.component_dirs) if not specified (DDPM only)
    - model.cond_dim from len(data.condition_columns) if not specified
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configuration dictionary with auto-completed values
    """
    config = config.copy()  # Don't modify original
    
    # Get model and data sections
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    model_type = model_cfg.get('type', 'ddpm')
    
    # Auto-calculate channels from component_dirs (DDPM, Flow Matching, VQ-VAE, and WGAN-GP, not MDM)
    if 'model' in config and model_type in ['ddpm', 'flow_matching', 'vqvae', 'wgan_gp']:
        if 'channels' not in model_cfg or model_cfg['channels'] is None:
            component_dirs = data_cfg.get('component_dirs', [])
            if component_dirs:
                model_cfg['channels'] = len(component_dirs)
                print(f"Auto-calculated model.channels = {len(component_dirs)} from component_dirs")
            else:
                raise ValueError(
                    "Cannot auto-calculate model.channels: data.component_dirs not specified"
                )
    
    # Auto-calculate total_latent_dim for WGAN-GP
    if 'model' in config and model_type == 'wgan_gp':
        if 'total_latent_dim' not in model_cfg or model_cfg['total_latent_dim'] is None:
            channels = model_cfg.get('channels')
            latent_dim_per_component = model_cfg.get('latent_dim_per_component', 4)
            if channels:
                model_cfg['total_latent_dim'] = channels * latent_dim_per_component
                print(f"Auto-calculated model.total_latent_dim = {channels * latent_dim_per_component} "
                      f"(channels={channels} × latent_dim_per_component={latent_dim_per_component})")
            else:
                raise ValueError(
                    "Cannot auto-calculate model.total_latent_dim: model.channels not specified"
                )
    
    # Auto-calculate cond_dim from condition_columns
    if 'model' in config:
        model_cfg = config['model']
        if 'cond_dim' not in model_cfg or model_cfg['cond_dim'] is None:
            condition_columns = data_cfg.get('condition_columns', [])
            if condition_columns:
                model_cfg['cond_dim'] = len(condition_columns)
                print(f"Auto-calculated model.cond_dim = {len(condition_columns)} from condition_columns")
            else:
                # For unconditional models, cond_dim can be 0
                if model_cfg.get('cond_drop_prob', 0.0) == 0.0:
                    model_cfg['cond_dim'] = 0
                    print("Auto-calculated model.cond_dim = 0 (unconditional model)")
                else:
                    raise ValueError(
                        "Cannot auto-calculate model.cond_dim: data.condition_columns not specified"
                    )
    
    return config


def validate_config(config: Dict[str, Any], model_type: Optional[str] = None) -> None:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        model_type: Expected model type ('ddpm' or 'mdm'), if None will try to infer
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Infer model type if not provided
    if model_type is None:
        model_type = config.get('model', {}).get('type', 'ddpm')
    
    # Validate model section
    if 'model' not in config:
        raise ValueError("Missing 'model' section in configuration")
    
    model_cfg = config['model']
    
    # Validate required model parameters (model-specific)
    if model_type == 'mmvaeplus':
        # MMVAE+ specific parameters
        required_mmvae_params = ['latent_dim_w', 'latent_dim_z', 'latent_dim_u', 'nf', 'nf_max']
        for param in required_mmvae_params:
            if param not in model_cfg:
                raise ValueError(f"Missing required model parameter: {param} (required for MMVAE+)")
        # Validate latent_dim_u = latent_dim_w + latent_dim_z
        if model_cfg.get('latent_dim_u') != model_cfg.get('latent_dim_w', 0) + model_cfg.get('latent_dim_z', 0):
            raise ValueError(
                f"model.latent_dim_u ({model_cfg.get('latent_dim_u')}) must equal "
                f"latent_dim_w ({model_cfg.get('latent_dim_w')}) + latent_dim_z ({model_cfg.get('latent_dim_z')})"
            )
    else:
        # Common parameters for diffusion models
        required_model_params = ['image_size', 'dim', 'dim_mults']
        for param in required_model_params:
            if param not in model_cfg:
                raise ValueError(f"Missing required model parameter: {param}")
    
    # Validate timesteps only for diffusion models
    if model_type in ['ddpm', 'flow_matching', 'mdm']:
        if 'timesteps' not in model_cfg:
            raise ValueError(f"Missing required model parameter: timesteps (required for {model_type})")
    
    # Validate data section
    if 'data' not in config:
        raise ValueError("Missing 'data' section in configuration")
    
    data_cfg = config['data']
    
    # Validate required data parameters
    required_data_params = ['root_dir', 'condition_csv']
    for param in required_data_params:
        if param not in data_cfg:
            raise ValueError(f"Missing required data parameter: {param}")
    
    # Model-specific validations
    if model_type == 'ddpm':
        # DDPM requires component_dirs
        if 'component_dirs' not in data_cfg:
            raise ValueError("DDPM requires data.component_dirs")
        
        # Validate channels matches component_dirs length
        if 'channels' in model_cfg and 'component_dirs' in data_cfg:
            if model_cfg['channels'] != len(data_cfg['component_dirs']):
                raise ValueError(
                    f"model.channels ({model_cfg['channels']}) does not match "
                    f"len(data.component_dirs) ({len(data_cfg['component_dirs'])})"
                )
    
    elif model_type == 'flow_matching':
        # Flow Matching requires component_dirs (similar to DDPM)
        if 'component_dirs' not in data_cfg:
            raise ValueError("Flow Matching requires data.component_dirs")
        
        # Validate channels matches component_dirs length
        if 'channels' in model_cfg and 'component_dirs' in data_cfg:
            if model_cfg['channels'] != len(data_cfg['component_dirs']):
                raise ValueError(
                    f"model.channels ({model_cfg['channels']}) does not match "
                    f"len(data.component_dirs) ({len(data_cfg['component_dirs'])})"
                )
        
        # Validate solver if present
        if 'solver' in model_cfg:
            valid_solvers = ['euler', 'heun', 'rk4']
            if model_cfg['solver'] not in valid_solvers:
                raise ValueError(
                    f"model.solver must be one of {valid_solvers}, got {model_cfg['solver']}"
                )
    
    elif model_type == 'vqvae':
        # VQ-VAE requires component_dirs (similar to DDPM)
        if 'component_dirs' not in data_cfg:
            raise ValueError("VQ-VAE requires data.component_dirs")
        
        # Validate channels matches component_dirs length
        if 'channels' in model_cfg and 'component_dirs' in data_cfg:
            if model_cfg['channels'] != len(data_cfg['component_dirs']):
                raise ValueError(
                    f"model.channels ({model_cfg['channels']}) does not match "
                    f"len(data.component_dirs) ({len(data_cfg['component_dirs'])})"
                )
        
        # Validate VQ-VAE specific parameters
        if 'latent_dim' not in model_cfg:
            raise ValueError("VQ-VAE requires model.latent_dim")
        if 'num_embeddings' not in model_cfg:
            raise ValueError("VQ-VAE requires model.num_embeddings")
    
    elif model_type == 'wgan_gp':
        # WGAN-GP requires component_dirs
        if 'component_dirs' not in data_cfg:
            raise ValueError("WGAN-GP requires data.component_dirs")
        
        # Validate channels matches component_dirs length (only if channels is set, None is OK for auto-completion)
        if 'channels' in model_cfg and model_cfg['channels'] is not None and 'component_dirs' in data_cfg:
            if model_cfg['channels'] != len(data_cfg['component_dirs']):
                raise ValueError(
                    f"model.channels ({model_cfg['channels']}) does not match "
                    f"len(data.component_dirs) ({len(data_cfg['component_dirs'])})"
                )
        
        # Validate WGAN-GP specific parameters
        if 'latent_dim_per_component' not in model_cfg:
            raise ValueError("WGAN-GP requires model.latent_dim_per_component")
        
        if 'lambda_gp' not in model_cfg:
            raise ValueError("WGAN-GP requires model.lambda_gp")
        
        if 'n_critic' not in model_cfg:
            raise ValueError("WGAN-GP requires model.n_critic")
    
    elif model_type == 'mdm':
        # MDM requires condition_columns (but not component_dirs)
        if 'condition_columns' not in data_cfg:
            raise ValueError("MDM requires data.condition_columns")
        
        # MDM doesn't use channels, it uses num_classes
        if 'num_classes' not in model_cfg:
            raise ValueError("MDM requires model.num_classes")
    
    elif model_type == 'mmvaeplus':
        # MMVAE+ requires component_dirs
        if 'component_dirs' not in data_cfg:
            raise ValueError("MMVAE+ requires data.component_dirs")
        
        # Validate component count (should be 5)
        if len(data_cfg['component_dirs']) != 5:
            raise ValueError(
                f"MMVAE+ requires exactly 5 components, got {len(data_cfg['component_dirs'])}"
            )
    
    # Validate condition_columns if present
    if 'condition_columns' in data_cfg:
        if not isinstance(data_cfg['condition_columns'], list):
            raise ValueError("data.condition_columns must be a list")
        if len(data_cfg['condition_columns']) == 0:
            raise ValueError("data.condition_columns cannot be empty")
    
    # Validate cond_dim matches condition_columns length if both present
    if 'cond_dim' in model_cfg and 'condition_columns' in data_cfg:
        if model_cfg['cond_dim'] != len(data_cfg['condition_columns']):
            raise ValueError(
                f"model.cond_dim ({model_cfg['cond_dim']}) does not match "
                f"len(data.condition_columns) ({len(data_cfg['condition_columns'])})"
            )
    
    # Validate training section
    if 'training' not in config:
        raise ValueError("Missing 'training' section in configuration")
    
    training_cfg = config['training']
    
    # Common required parameters
    required_training_params = ['epochs', 'batch_size']
    for param in required_training_params:
        if param not in training_cfg:
            raise ValueError(f"Missing required training parameter: {param}")
    
    # Learning rate validation (model-specific)
    if model_type == 'vqvae':
        # VQ-VAE uses lr_vqvae and lr_prior
        if 'lr_vqvae' not in training_cfg:
            raise ValueError("Missing required training parameter: lr_vqvae (required for VQ-VAE)")
        if training_cfg.get('train_prior_after_epoch', 0) > 0 and 'lr_prior' not in training_cfg:
            raise ValueError("Missing required training parameter: lr_prior (required for VQ-VAE with prior)")
    elif model_type == 'wgan_gp':
        # WGAN-GP uses lr_generator and lr_critic
        if 'lr_generator' not in training_cfg:
            raise ValueError("Missing required training parameter: lr_generator (required for WGAN-GP)")
        if 'lr_critic' not in training_cfg:
            raise ValueError("Missing required training parameter: lr_critic (required for WGAN-GP)")
    elif model_type == 'mmvaeplus':
        # MMVAE+ uses lr
        if 'lr' not in training_cfg:
            raise ValueError("Missing required training parameter: lr (required for MMVAE+)")
        # Validate objective
        if 'objective' in training_cfg:
            valid_objectives = ['elbo', 'dreg']
            if training_cfg['objective'] not in valid_objectives:
                raise ValueError(
                    f"training.objective must be one of {valid_objectives}, got {training_cfg['objective']}"
                )
    else:
        # Other models use lr
        if 'lr' not in training_cfg:
            raise ValueError(f"Missing required training parameter: lr (required for {model_type})")


def get_project_root() -> Path:
    """Get project root directory."""
    _THIS_FILE = Path(__file__).resolve()
    # src_new/utils/config.py -> src_new -> project root
    return _THIS_FILE.parent.parent.parent


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to base_dir or project root.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_dir: Base directory for relative paths (default: project root)
    
    Returns:
        Absolute Path object
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base_dir is None:
        base_dir = get_project_root()
    
    return (base_dir / path).resolve()

