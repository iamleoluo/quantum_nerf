#!/usr/bin/env python3
"""
NeRF PyTorch Implementation - From Scratch
Quantum-Ready Architecture

This implementation is designed to be:
1. Educational and easy to understand
2. Modular for quantum integration
3. Based on the original NeRF paper
4. Ready to run with minimal setup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from typing import Tuple, Optional, Dict, Any
import imageio
from PIL import Image

# cuda check
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

class PositionalEncoder:
    """
    Positional encoding using sinusoidal functions
    This module can be replaced with quantum encoding in the future
    """
    
    def __init__(self, input_dims: int = 3, max_freq_log2: int = 10, 
                 num_freqs: int = 10, include_input: bool = True):
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.periodic_fns = [torch.sin, torch.cos]
        
        # Create frequency bands
        freq_bands = 2.**torch.linspace(0., max_freq_log2, steps=num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # Calculate output dimensions
        out_dim = 0
        if include_input:
            out_dim += input_dims
        out_dim += input_dims * len(self.periodic_fns) * num_freqs
        self.out_dim = out_dim
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Simple buffer registration for frequency bands"""
        setattr(self, name, tensor)
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates
        
        Args:
            inputs: [..., input_dims] coordinates
            
        Returns:
            [..., out_dim] encoded coordinates
        """
        outputs = []
        
        if self.include_input:
            outputs.append(inputs)
        
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                outputs.append(p_fn(inputs * freq))
        
        return torch.cat(outputs, dim=-1)
    
    # Quantum integration point
    def quantum_encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for future quantum positional encoding
        Could implement quantum Fourier transforms or variational circuits
        """
        # TODO: Implement quantum encoding
        # For now, fallback to classical encoding
        return self.encode(inputs)


class NeRFNetwork(nn.Module):
    """
    NeRF neural network with quantum-ready architecture
    
    The network predicts color and density for 3D points
    """
    
    def __init__(self, 
                 pos_encode_dim: int = 63,
                 dir_encode_dim: int = 27,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 skip_connections: list = None):
        super().__init__()
        
        if skip_connections is None:
            skip_connections = [4]
        
        self.pos_encode_dim = pos_encode_dim
        self.dir_encode_dim = dir_encode_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        # Position processing layers
        self.pos_layers = nn.ModuleList()
        self.pos_layers.append(nn.Linear(pos_encode_dim, hidden_dim))
        
        for i in range(1, num_layers):
            if i in skip_connections:
                self.pos_layers.append(nn.Linear(hidden_dim + pos_encode_dim, hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Density prediction head
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature extraction for color
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Color prediction layers (view-dependent)
        self.color_layers = nn.ModuleList([
            nn.Linear(hidden_dim + dir_encode_dim, hidden_dim // 2)
        ])
        self.rgb_head = nn.Linear(hidden_dim // 2, 3)
        
        # Quantum layer placeholders for future integration
        self.quantum_layers = nn.ModuleList()
        self.use_quantum = False
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the NeRF network
        
        Args:
            pos_encoded: [batch, pos_encode_dim] encoded positions
            dir_encoded: [batch, dir_encode_dim] encoded directions
            
        Returns:
            rgb: [batch, 3] predicted colors
            density: [batch, 1] predicted densities
        """
        # Process positions through MLP
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            h = layer(h)
            h = F.relu(h)
            
            # Apply skip connections after processing the layer
            # Skip connections concatenate the original input to the current features
            if (i + 1) in self.skip_connections and (i + 1) < len(self.pos_layers):
                h = torch.cat([h, pos_encoded], dim=-1)
        
        # Predict density (view-independent)
        density = F.relu(self.density_head(h))
        
        # Extract features for color prediction
        features = self.feature_layer(h)
        
        # Combine features with viewing direction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        
        # Process through color layers
        for layer in self.color_layers:
            color_input = F.relu(layer(color_input))
        
        # Predict RGB color
        rgb = torch.sigmoid(self.rgb_head(color_input))
        
        return rgb, density
    
    def quantum_forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Placeholder for quantum-enhanced forward pass
        """
        if self.use_quantum and len(self.quantum_layers) > 0:
            # TODO: Implement quantum layer processing
            pass
        
        # Fallback to classical processing
        return self.forward(pos_encoded, dir_encoded)


class VolumeRenderer:
    """
    Volume rendering implementation
    Supports both classical and quantum sampling strategies
    """
    
    def __init__(self, near: float = 2.0, far: float = 6.0, 
                 n_samples: int = 64, n_importance: int = 128):
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_importance = n_importance
    
    def sample_points_along_ray(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                               near: float, far: float, n_samples: int, 
                               perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays using stratified sampling
        
        Args:
            rays_o: [batch, 3] ray origins
            rays_d: [batch, 3] ray directions
            near, far: near and far bounds
            n_samples: number of samples per ray
            perturb: whether to add noise to sampling
            
        Returns:
            pts: [batch, n_samples, 3] sampled points
            z_vals: [batch, n_samples] depths along rays
        """
        # Create evenly spaced samples
        t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand([rays_o.shape[0], n_samples])
        
        # Add perturbation for training (stratified sampling)
        if perturb:
            # Get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            
            # Stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=rays_o.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Calculate 3D points along rays
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return pts, z_vals
    
    def volume_render(self, rgb: torch.Tensor, density: torch.Tensor, 
                     z_vals: torch.Tensor, rays_d: torch.Tensor, 
                     white_bkgd: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform volume rendering
        
        Args:
            rgb: [batch, n_samples, 3] color values
            density: [batch, n_samples, 1] density values
            z_vals: [batch, n_samples] depth values
            rays_d: [batch, 3] ray directions
            white_bkgd: whether to use white background
            
        Returns:
            rgb_map: [batch, 3] rendered colors
            depth_map: [batch] depth values
            weights: [batch, n_samples] importance weights
        """
        # Calculate distances between adjacent samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)
        
        # Multiply each distance by the norm of its ray direction
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Calculate alpha values (probability of ray termination)
        alpha = 1. - torch.exp(-density[..., 0] * dists)
        
        # Calculate transmittance (cumulative product of (1 - alpha))
        transmittance = torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
        )[:, :-1]
        
        # Calculate importance weights
        weights = alpha * transmittance
        
        # Composite final color
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        
        # Calculate depth map
        depth_map = torch.sum(weights * z_vals, -1)
        
        # Add white background if specified
        if white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        return rgb_map, depth_map, weights
    
    def quantum_sample_points(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                             near: float, far: float, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Placeholder for quantum sampling strategies
        Could implement quantum random number generation or quantum optimization
        """
        # TODO: Implement quantum sampling
        # For now, fallback to classical sampling
        return self.sample_points_along_ray(rays_o, rays_d, near, far, n_samples)


class NeRFTrainer:
    """
    Training pipeline with support for quantum enhancements
    """
    
    def __init__(self, model: NeRFNetwork, pos_encoder: PositionalEncoder, 
                 dir_encoder: PositionalEncoder, renderer: VolumeRenderer, 
                 config: Dict[str, Any]):
        self.model = model
        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder
        self.renderer = renderer
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Quantum components (placeholders)
        self.quantum_optimizer = None
        self.quantum_loss = None
    
    def validate(self, val_data: Dict[str, torch.Tensor], 
                num_val_images: int = 5) -> Tuple[float, float]:
        """
        Validate the model on validation data
        
        Args:
            val_data: validation dataset
            num_val_images: number of validation images to evaluate
            
        Returns:
            average_loss: average validation loss
            average_psnr: average validation PSNR
        """
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_evaluated = 0
        
        with torch.no_grad():
            # Randomly sample validation images
            val_indices = torch.randperm(len(val_data['images']))[:num_val_images]
            
            for idx in val_indices:
                target_img = val_data['images'][idx]
                pose = val_data['poses'][idx]
                
                # Create rays for this image
                rays_o, rays_d = create_rays(
                    val_data['height'], val_data['width'], val_data['focal'], pose
                )
                
                # Render the image
                rendered_img = self.render_image(rays_o, rays_d, chunk_size=512)
                rendered_img = rendered_img.reshape(val_data['height'], val_data['width'], 3)
                
                # Calculate loss
                loss = self.mse_loss(rendered_img, target_img)
                psnr = -10. * torch.log10(loss)
                
                total_loss += loss.item()
                total_psnr += psnr.item()
                num_evaluated += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_evaluated
        avg_psnr = total_psnr / num_evaluated
        
        return avg_loss, avg_psnr

    def train_step(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                  target_rgb: torch.Tensor) -> Tuple[float, float]:
        """
        Perform one training step
        
        Args:
            rays_o: [batch, 3] ray origins
            rays_d: [batch, 3] ray directions
            target_rgb: [batch, 3] target colors
            
        Returns:
            loss: scalar loss value
            psnr: peak signal-to-noise ratio
        """
        # Move to device
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)
        target_rgb = target_rgb.to(self.device)
        
        # Sample points along rays
        pts, z_vals = self.renderer.sample_points_along_ray(
            rays_o, rays_d, 
            self.renderer.near, self.renderer.far, 
            self.renderer.n_samples, 
            perturb=True
        )
        
        # Get viewing directions
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None].expand(pts.shape)
        
        # Flatten for processing
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = viewdirs.reshape(-1, 3)
        
        # Encode positions and directions
        pts_encoded = self.pos_encoder.encode(pts_flat)
        dirs_encoded = self.dir_encoder.encode(dirs_flat)
        
        # Forward pass through network
        rgb, density = self.model(pts_encoded, dirs_encoded)
        
        # Reshape back to ray samples
        rgb = rgb.reshape(*pts.shape[:-1], 3)
        density = density.reshape(*pts.shape[:-1], 1)
        
        # Volume rendering
        rgb_map, depth_map, weights = self.renderer.volume_render(
            rgb, density, z_vals, rays_d, white_bkgd=True
        )
        
        # Calculate loss
        loss = self.mse_loss(rgb_map, target_rgb)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate PSNR
        psnr = -10. * torch.log10(loss)
        
        return loss.item(), psnr.item()
    
    def render_image(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                    chunk_size: int = 1024) -> torch.Tensor:
        """
        Render a full image using the trained model
        
        Args:
            rays_o: [H*W, 3] ray origins
            rays_d: [H*W, 3] ray directions
            chunk_size: process rays in chunks to avoid OOM
            
        Returns:
            rgb_map: [H*W, 3] rendered colors
        """
        self.model.eval()
        
        rgb_maps = []
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i:i+chunk_size].to(self.device)
                chunk_rays_d = rays_d[i:i+chunk_size].to(self.device)
                
                # Sample points
                pts, z_vals = self.renderer.sample_points_along_ray(
                    chunk_rays_o, chunk_rays_d,
                    self.renderer.near, self.renderer.far,
                    self.renderer.n_samples,
                    perturb=False
                )
                
                # Get viewing directions
                viewdirs = chunk_rays_d / torch.norm(chunk_rays_d, dim=-1, keepdim=True)
                viewdirs = viewdirs[:, None].expand(pts.shape)
                
                # Flatten and encode
                pts_flat = pts.reshape(-1, 3)
                dirs_flat = viewdirs.reshape(-1, 3)
                
                pts_encoded = self.pos_encoder.encode(pts_flat)
                dirs_encoded = self.dir_encoder.encode(dirs_flat)
                
                # Forward pass
                rgb, density = self.model(pts_encoded, dirs_encoded)
                
                # Reshape and render
                rgb = rgb.reshape(*pts.shape[:-1], 3)
                density = density.reshape(*pts.shape[:-1], 1)
                
                rgb_map, _, _ = self.renderer.volume_render(
                    rgb, density, z_vals, chunk_rays_d, white_bkgd=True
                )
                
                rgb_maps.append(rgb_map.cpu())
        
        self.model.train()
        return torch.cat(rgb_maps, dim=0)


def create_rays(height: int, width: int, focal: float, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create rays for a given camera pose
    
    Args:
        height, width: image dimensions
        focal: focal length
        pose: [3, 4] camera pose matrix
        
    Returns:
        rays_o: [H*W, 3] ray origins
        rays_d: [H*W, 3] ray directions
    """
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.linspace(0, width-1, width),
        torch.linspace(0, height-1, height),
        indexing='xy'
    )
    
    # Convert to camera coordinates
    dirs = torch.stack([
        (i - width * 0.5) / focal,
        -(j - height * 0.5) / focal,
        -torch.ones_like(i)
    ], -1)
    
    # Rotate ray directions from camera to world frame
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
    
    # Ray origins are the camera position
    rays_o = pose[:3, -1].expand(rays_d.shape)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def load_nerf_dataset(data_dir: str, split: str = 'train', 
                     downsample_factor: int = 1, white_bkgd: bool = True) -> Dict[str, torch.Tensor]:
    """
    Load NeRF dataset from the standard format
    
    Args:
        data_dir: path to the dataset directory
        split: 'train', 'val', or 'test'
        downsample_factor: factor to downsample images (1 = no downsampling)
        white_bkgd: whether to use white background for transparent images
        
    Returns:
        Dictionary containing images, poses, intrinsics, and metadata
    """
    # Load transforms file
    transforms_file = os.path.join(data_dir, f'transforms_{split}.json')
    
    if not os.path.exists(transforms_file):
        raise FileNotFoundError(f"Transforms file not found: {transforms_file}")
    
    with open(transforms_file, 'r') as f:
        meta = json.load(f)
    
    # Extract camera parameters
    camera_angle_x = meta['camera_angle_x']
    
    # Load images and poses
    images = []
    poses = []
    
    print(f"Loading {split} dataset from {data_dir}...")
    
    for frame in tqdm(meta['frames'], desc=f"Loading {split} images"):
        # Construct image path
        image_path = frame['file_path']
        if not image_path.endswith('.png'):
            image_path += '.png'
        
        # Handle relative paths
        if image_path.startswith('./'):
            image_path = image_path[2:]
        
        full_image_path = os.path.join(data_dir, image_path)
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        # Load image
        img = Image.open(full_image_path)
        img = np.array(img)
        
        # Handle RGBA images
        if img.shape[-1] == 4:
            # Extract alpha channel
            alpha = img[..., 3:4] / 255.0
            rgb = img[..., :3] / 255.0
            
            if white_bkgd:
                # Composite with white background
                img = rgb * alpha + (1.0 - alpha)
            else:
                # Keep alpha channel
                img = np.concatenate([rgb, alpha], axis=-1)
        else:
            # RGB image
            img = img[..., :3] / 255.0
        
        # Downsample if needed
        if downsample_factor > 1:
            h, w = img.shape[:2]
            new_h, new_w = h // downsample_factor, w // downsample_factor
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((new_w, new_h)))
            img = img.astype(np.float32) / 255.0
        
        images.append(img)
        
        # Extract pose matrix
        pose = np.array(frame['transform_matrix'])
        poses.append(pose)
    
    # Convert to tensors
    images = torch.from_numpy(np.stack(images)).float()
    poses = torch.from_numpy(np.stack(poses)).float()
    
    # Calculate focal length from camera angle
    height, width = images.shape[1:3]
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    
    # Adjust focal length for downsampling
    focal = focal / downsample_factor
    
    print(f"Loaded {len(images)} images of size {height}x{width}")
    print(f"Focal length: {focal:.2f}")
    
    return {
        'images': images,
        'poses': poses,
        'height': height,
        'width': width,
        'focal': focal,
        'camera_angle_x': camera_angle_x
    }


def generate_synthetic_data(n_images: int = 100) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic training data for demonstration
    In practice, you would load real images and camera poses
    """
    # Create simple camera poses in a circle
    angles = torch.linspace(0, 2*np.pi, n_images)
    radius = 4.0
    
    poses = []
    for angle in angles:
        # Camera position
        cam_pos = torch.tensor([
            radius * torch.cos(angle),
            0.0,
            radius * torch.sin(angle)
        ])
        
        # Look at origin
        look_at = torch.tensor([0., 0., 0.])
        up = torch.tensor([0., 1., 0.])
        
        # Create camera-to-world matrix
        z_axis = F.normalize(cam_pos - look_at, dim=0)
        x_axis = F.normalize(torch.cross(up, z_axis, dim=0), dim=0)
        y_axis = torch.cross(z_axis, x_axis, dim=0)
        
        pose = torch.stack([x_axis, y_axis, z_axis, cam_pos], dim=1)
        poses.append(pose)
    
    poses = torch.stack(poses)
    
    # Create dummy images (in practice, load real images)
    height, width = 100, 100
    images = torch.rand(n_images, height, width, 3)
    
    return {
        'images': images,
        'poses': poses,
        'height': height,
        'width': width,
        'focal': 50.0
    }


def main():
    """
    Main training loop
    """
    # Configuration
    config = {
        'learning_rate': 5e-4,
        'batch_size': 1024,
        'num_epochs': 10000,
        'pos_encode_freqs': 10,
        'dir_encode_freqs': 4,
        'log_every': 100,
        'save_every': 1000,
        'data_dir': 'data/synthetic',  # Path to the real dataset
        'downsample_factor': 4,  # Downsample images for faster training
        'use_real_data': True  # Switch between real and synthetic data
    }
    
    print("🚀 Starting NeRF Training (Quantum-Ready Architecture)")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load data
    if config['use_real_data']:
        print("📁 Loading real NeRF dataset...")
        data = load_nerf_dataset(
            data_dir=config['data_dir'],
            split='train',
            downsample_factor=config['downsample_factor'],
            white_bkgd=True
        )
        
        # Also load validation data for evaluation
        val_data = load_nerf_dataset(
            data_dir=config['data_dir'],
            split='val',
            downsample_factor=config['downsample_factor'],
            white_bkgd=True
        )
        print(f"📊 Training images: {len(data['images'])}")
        print(f"📊 Validation images: {len(val_data['images'])}")
    else:
        print("🎲 Generating synthetic data...")
        data = generate_synthetic_data(n_images=50)
        val_data = None
    
    # Initialize encoders
    pos_encoder = PositionalEncoder(
        input_dims=3, 
        num_freqs=config['pos_encode_freqs']
    )
    dir_encoder = PositionalEncoder(
        input_dims=3, 
        num_freqs=config['dir_encode_freqs']
    )
    
    # Initialize network
    model = NeRFNetwork(
        pos_encode_dim=pos_encoder.out_dim,
        dir_encode_dim=dir_encoder.out_dim
    )
    
    # Initialize renderer
    renderer = VolumeRenderer()
    
    # Initialize trainer
    trainer = NeRFTrainer(model, pos_encoder, dir_encoder, renderer, config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Positional encoding dim: {pos_encoder.out_dim}")
    print(f"Directional encoding dim: {dir_encoder.out_dim}")
    
    # Training loop
    best_val_psnr = 0.0
    
    for epoch in tqdm(range(config['num_epochs']), desc="Training"):
        # Sample random image
        img_idx = torch.randint(0, data['images'].shape[0], (1,)).item()
        target_img = data['images'][img_idx]
        pose = data['poses'][img_idx]
        
        # Create rays for this image
        rays_o, rays_d = create_rays(
            data['height'], data['width'], data['focal'], pose
        )
        
        # Sample random rays from the image
        ray_indices = torch.randperm(rays_o.shape[0])[:config['batch_size']]
        batch_rays_o = rays_o[ray_indices]
        batch_rays_d = rays_d[ray_indices]
        
        # Get target colors (ensure RGB format)
        if target_img.shape[-1] == 4:
            target_img = target_img[..., :3]  # Remove alpha channel if present
        target_rgb = target_img.reshape(-1, 3)[ray_indices]
        
        # Training step
        loss, psnr = trainer.train_step(batch_rays_o, batch_rays_d, target_rgb)
        
        # Logging and validation
        if epoch % config['log_every'] == 0:
            log_msg = f"Epoch {epoch:5d}: Train Loss = {loss:.6f}, Train PSNR = {psnr:.2f}"
            
            # Run validation if we have validation data
            if val_data is not None and epoch % (config['log_every'] * 2) == 0:
                val_loss, val_psnr = trainer.validate(val_data, num_val_images=3)
                log_msg += f", Val Loss = {val_loss:.6f}, Val PSNR = {val_psnr:.2f}"
                
                # Save best model
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'epoch': epoch,
                        'config': config,
                        'val_psnr': val_psnr
                    }, 'nerf_best_model.pth')
                    log_msg += " 🌟 (Best!)"
            
            print(log_msg)
        
        # Save checkpoint
        if epoch % config['save_every'] == 0 and epoch > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, f'nerf_checkpoint_{epoch}.pth')
            print(f"💾 Saved checkpoint at epoch {epoch}")
    
    print("✅ Training completed!")
    print(f"🏆 Best validation PSNR: {best_val_psnr:.2f}")
    
    # Render test images
    print("🎨 Rendering test images...")
    
    # Use validation data for testing if available, otherwise use training data
    test_data = val_data if val_data is not None else data
    
    # Render a few test images
    num_test_renders = min(3, len(test_data['images']))
    for i in range(num_test_renders):
        test_pose = test_data['poses'][i]
        rays_o, rays_d = create_rays(test_data['height'], test_data['width'], test_data['focal'], test_pose)
        
        rendered_img = trainer.render_image(rays_o, rays_d)
        rendered_img = rendered_img.reshape(test_data['height'], test_data['width'], 3)
        
        # Save rendered image
        rendered_filename = f'rendered_image_{i:03d}.png'
        imageio.imwrite(rendered_filename, (rendered_img * 255).numpy().astype(np.uint8))
        
        # Also save ground truth for comparison
        if i < len(test_data['images']):
            gt_img = test_data['images'][i]
            gt_filename = f'ground_truth_{i:03d}.png'
            imageio.imwrite(gt_filename, (gt_img * 255).numpy().astype(np.uint8))
        
        print(f"💾 Saved rendered image: {rendered_filename}")
    
    # Save final model
    torch.save(model.state_dict(), 'nerf_final_model.pth')
    print("💾 Saved final model as 'nerf_final_model.pth'")
    
    # Print summary
    print("\n📊 Training Summary:")
    print(f"   • Dataset: {'Real NeRF data' if config['use_real_data'] else 'Synthetic data'}")
    print(f"   • Training images: {len(data['images'])}")
    if val_data is not None:
        print(f"   • Validation images: {len(val_data['images'])}")
    print(f"   • Image resolution: {data['height']}x{data['width']}")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Best validation PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
