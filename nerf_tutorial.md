# Building NeRF PyTorch From Scratch 🚀

## Table of Contents
1. [Understanding NeRF Conceptually](#understanding-nerf)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Step-by-Step Implementation](#implementation)
4. [Training Pipeline](#training)
5. [Quantum Integration Architecture](#quantum-ready)

---

## Understanding NeRF Conceptually {#understanding-nerf}

**What is NeRF?**
- **Neural Radiance Fields** represents 3D scenes as continuous functions
- Input: 3D position (x,y,z) + viewing direction (θ,φ) → Output: Color (RGB) + Density (σ)
- Creates photorealistic novel views from limited input images

**Key Insight:** Instead of storing 3D geometry explicitly, we learn a neural function that maps spatial coordinates to scene properties.

---

## Mathematical Foundation {#mathematical-foundation}

### Core Equation: Volume Rendering
```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
```
Where:
- `C(r)`: Final pixel color along ray r
- `T(t)`: Transmittance (how much light reaches point t)
- `σ(r(t))`: Density at point r(t)
- `c(r(t), d)`: Color at point r(t) viewing from direction d

### Discrete Approximation
```
C(r) = Σ Ti · (1 - exp(-σi·δi)) · ci
Ti = exp(-Σ σj·δj) for j < i
```

---

## Step-by-Step Implementation {#implementation}

### Step 1: Core Architecture Design

```python
# Base architecture that's quantum-ready
class NeRFCore:
    """
    Modular NeRF architecture designed for extensibility
    - Classical components can be swapped with quantum versions
    - Clean interfaces for hybrid classical-quantum models
    """
    
    def __init__(self, config):
        self.encoder = self._create_encoder(config)      # Position encoder
        self.decoder = self._create_decoder(config)      # Neural network
        self.renderer = self._create_renderer(config)    # Volume renderer
        
        # Quantum integration points (placeholders for now)
        self.quantum_processor = None  # Future: quantum position encoding
        self.quantum_sampler = None    # Future: quantum ray sampling
```

### Step 2: Positional Encoding Module

```python
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoder:
    """
    Encode 3D positions using sinusoidal functions
    This is where quantum encoding could be integrated later
    """
    
    def __init__(self, input_dims=3, max_freq_log2=10, num_freqs=10):
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.periodic_fns = [torch.sin, torch.cos]
        
        # Create frequency bands
        self.freq_bands = 2.**torch.linspace(0., max_freq_log2, steps=num_freqs)
        
        # Calculate output dimensions
        self.out_dim = input_dims * (1 + 2 * num_freqs)
    
    def encode(self, inputs):
        """
        Encode positions using sinusoidal functions
        
        Args:
            inputs: [..., input_dims] positions
        Returns:
            [..., out_dim] encoded positions
        """
        outputs = [inputs]  # Include original coordinates
        
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                outputs.append(p_fn(inputs * freq))
        
        return torch.cat(outputs, dim=-1)
    
    # Quantum integration point
    def quantum_encode(self, inputs):
        """
        Placeholder for quantum positional encoding
        Could use quantum Fourier transforms or variational circuits
        """
        # TODO: Implement quantum encoding
        return self.encode(inputs)  # Fallback to classical for now
```

### Step 3: Neural Network Architecture

```python
class NeRFNetwork(nn.Module):
    """
    Main NeRF neural network
    Designed with modularity for quantum layer integration
    """
    
    def __init__(self, 
                 pos_encode_dim=63,     # 3 + 3*2*10 positional encoding
                 dir_encode_dim=27,     # 3 + 3*2*4 directional encoding  
                 hidden_dim=256,
                 num_layers=8,
                 skip_connections=[4]):
        
        super().__init__()
        
        # Position processing layers
        self.pos_layers = nn.ModuleList()
        self.pos_layers.append(nn.Linear(pos_encode_dim, hidden_dim))
        
        for i in range(1, num_layers):
            if i in skip_connections:
                self.pos_layers.append(nn.Linear(hidden_dim + pos_encode_dim, hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Density prediction
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature extraction for color
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Color prediction (depends on viewing direction)
        self.color_layer = nn.Linear(hidden_dim + dir_encode_dim, hidden_dim // 2)
        self.rgb_head = nn.Linear(hidden_dim // 2, 3)
        
        self.skip_connections = skip_connections
        
        # Quantum layer placeholders
        self.quantum_layers = nn.ModuleList()  # For future quantum layers
    
    def forward(self, pos_encoded, dir_encoded):
        """
        Forward pass through NeRF network
        
        Args:
            pos_encoded: [batch, pos_encode_dim] encoded positions
            dir_encoded: [batch, dir_encode_dim] encoded directions
        
        Returns:
            rgb: [batch, 3] color values
            density: [batch, 1] density values
        """
        # Process positions through MLP
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            h = layer(h)
            h = torch.relu(h)
            
            # Skip connections
            if i in self.skip_connections:
                h = torch.cat([h, pos_encoded], dim=-1)
        
        # Predict density (doesn't depend on view direction)
        density = torch.relu(self.density_head(h))
        
        # Extract features for color prediction
        features = self.feature_layer(h)
        
        # Combine features with view direction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        color_features = torch.relu(self.color_layer(color_input))
        rgb = torch.sigmoid(self.rgb_head(color_features))
        
        return rgb, density
    
    def quantum_forward(self, pos_encoded, dir_encoded):
        """
        Placeholder for quantum-enhanced forward pass
        Could integrate quantum layers for specific computations
        """
        # TODO: Implement quantum layers
        return self.forward(pos_encoded, dir_encoded)
```

### Step 4: Ray Sampling and Rendering

```python
class VolumeRenderer:
    """
    Implements volume rendering equation
    Designed for both classical and quantum sampling strategies
    """
    
    def __init__(self, near=2.0, far=6.0, n_samples=64, n_importance=128):
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_importance = n_importance
    
    def sample_points_along_ray(self, rays_o, rays_d, near, far, n_samples, perturb=True):
        """
        Sample points along rays
        
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
        # Create depth samples
        t_vals = torch.linspace(0., 1., steps=n_samples)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand([rays_o.shape[0], n_samples])
        
        # Add perturbation for training
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand
        
        # Get 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return pts, z_vals
    
    def quantum_sample_points(self, rays_o, rays_d, near, far, n_samples):
        """
        Placeholder for quantum sampling strategies
        Could use quantum random number generation or optimization
        """
        # TODO: Implement quantum sampling
        return self.sample_points_along_ray(rays_o, rays_d, near, far, n_samples)
    
    def volume_render(self, rgb, density, z_vals, rays_d, white_bkgd=False):
        """
        Perform volume rendering given rgb and density values
        
        Args:
            rgb: [batch, n_samples, 3] color values
            density: [batch, n_samples, 1] density values
            z_vals: [batch, n_samples] depth values
            rays_d: [batch, 3] ray directions
            white_bkgd: whether to use white background
        
        Returns:
            rgb_map: [batch, 3] final pixel colors
            depth_map: [batch] depth values
            weights: [batch, n_samples] importance weights
        """
        # Calculate distances between adjacent samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)
        
        # Multiply each distance by the norm of its corresponding direction ray
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Calculate alpha values (probability of ray termination)
        alpha = 1. - torch.exp(-density[..., 0] * dists)
        
        # Calculate transmittance (probability of ray reaching each point)
        transmittance = torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1
        )[:, :-1]
        
        # Calculate importance weights
        weights = alpha * transmittance
        
        # Composite colors
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        
        # Calculate depth map
        depth_map = torch.sum(weights * z_vals, -1)
        
        # Add white background if specified
        if white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        return rgb_map, depth_map, weights
```

### Step 5: Training Pipeline

```python
class NeRFTrainer:
    """
    Training pipeline with modular design for quantum integration
    """
    
    def __init__(self, model, encoder, renderer, config):
        self.model = model
        self.encoder = encoder
        self.renderer = renderer
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Quantum training components (placeholders)
        self.quantum_optimizer = None  # Future: quantum optimization
        self.quantum_loss = None       # Future: quantum loss functions
    
    def train_step(self, rays_o, rays_d, target_rgb):
        """
        Single training step
        
        Args:
            rays_o: [batch, 3] ray origins
            rays_d: [batch, 3] ray directions  
            target_rgb: [batch, 3] target colors
        
        Returns:
            loss: scalar loss value
            psnr: peak signal-to-noise ratio
        """
        # Sample points along rays
        pts, z_vals = self.renderer.sample_points_along_ray(
            rays_o, rays_d, 
            self.renderer.near, self.renderer.far, 
            self.renderer.n_samples
        )
        
        # Get viewing directions
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None].expand(pts.shape)
        
        # Flatten points and directions
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = viewdirs.reshape(-1, 3)
        
        # Encode positions and directions
        pts_encoded = self.encoder.encode(pts_flat)
        dirs_encoded = self.encoder.encode(dirs_flat)
        
        # Run through network
        rgb, density = self.model(pts_encoded, dirs_encoded)
        
        # Reshape back
        rgb = rgb.reshape(*pts.shape[:-1], 3)
        density = density.reshape(*pts.shape[:-1], 1)
        
        # Volume rendering
        rgb_map, depth_map, weights = self.renderer.volume_render(
            rgb, density, z_vals, rays_d, white_bkgd=True
        )
        
        # Calculate loss
        loss = self.mse_loss(rgb_map, target_rgb)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate PSNR
        psnr = -10. * torch.log10(loss)
        
        return loss.item(), psnr.item()
    
    def quantum_train_step(self, rays_o, rays_d, target_rgb):
        """
        Placeholder for quantum-enhanced training
        Could integrate quantum optimization or quantum neural networks
        """
        # TODO: Implement quantum training enhancements
        return self.train_step(rays_o, rays_d, target_rgb)
```

---

## Training Pipeline {#training}

### Complete Training Loop

```python
def train_nerf():
    """
    Complete training pipeline
    """
    # Configuration
    config = {
        'learning_rate': 5e-4,
        'batch_size': 1024,
        'num_epochs': 200000,
        'pos_encode_freqs': 10,
        'dir_encode_freqs': 4
    }
    
    # Initialize components
    pos_encoder = PositionalEncoder(input_dims=3, num_freqs=config['pos_encode_freqs'])
    dir_encoder = PositionalEncoder(input_dims=3, num_freqs=config['dir_encode_freqs'])
    
    model = NeRFNetwork(
        pos_encode_dim=pos_encoder.out_dim,
        dir_encode_dim=dir_encoder.out_dim
    )
    
    renderer = VolumeRenderer()
    trainer = NeRFTrainer(model, pos_encoder, renderer, config)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Sample random rays (implement data loading)
        rays_o, rays_d, target_rgb = sample_random_rays(batch_size=config['batch_size'])
        
        # Training step
        loss, psnr = trainer.train_step(rays_o, rays_d, target_rgb)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}, PSNR = {psnr:.2f}")
            
        # Save checkpoint
        if epoch % 10000 == 0:
            torch.save(model.state_dict(), f'nerf_checkpoint_{epoch}.pth')
```

---

## Quantum Integration Architecture {#quantum-ready}

### Design Principles for Quantum Integration

```python
class QuantumNeRFArchitecture:
    """
    Architecture design that facilitates quantum integration
    
    Key Integration Points:
    1. Quantum Positional Encoding
    2. Quantum Neural Layers  
    3. Quantum Sampling Strategies
    4. Quantum Optimization
    """
    
    def __init__(self):
        # Classical components
        self.classical_encoder = PositionalEncoder()
        self.classical_network = NeRFNetwork()
        self.classical_renderer = VolumeRenderer()
        
        # Quantum components (to be implemented)
        self.quantum_encoder = None      # Quantum Fourier encoding
        self.quantum_layers = None       # Variational quantum circuits
        self.quantum_sampler = None      # Quantum-enhanced sampling
        self.quantum_optimizer = None    # Quantum approximate optimization
    
    def hybrid_forward(self, positions, directions):
        """
        Hybrid classical-quantum forward pass
        """
        # Option 1: Quantum encoding + Classical network
        if self.quantum_encoder:
            pos_encoded = self.quantum_encoder.encode(positions)
            dir_encoded = self.quantum_encoder.encode(directions)
        else:
            pos_encoded = self.classical_encoder.encode(positions)
            dir_encoded = self.classical_encoder.encode(directions)
        
        # Option 2: Classical encoding + Quantum layers
        if self.quantum_layers:
            return self.quantum_layers.forward(pos_encoded, dir_encoded)
        else:
            return self.classical_network(pos_encoded, dir_encoded)
```

### Future Quantum Enhancements

1. **Quantum Positional Encoding**
   - Use quantum Fourier transforms for position encoding
   - Potentially more efficient for high-dimensional spaces

2. **Variational Quantum Circuits**
   - Replace some classical layers with parameterized quantum circuits
   - Could provide quantum advantage for certain computations

3. **Quantum Sampling**
   - Quantum random number generation for ray sampling
   - Quantum-enhanced Monte Carlo methods

4. **Quantum Optimization**
   - Quantum Approximate Optimization Algorithm (QAOA)
   - Variational Quantum Eigensolver (VQE) for optimization

### Implementation Roadmap

```python
# Phase 1: Classical Implementation (Current)
classical_nerf = NeRFNetwork()

# Phase 2: Hybrid Architecture
hybrid_nerf = QuantumNeRFArchitecture()

# Phase 3: Full Quantum Integration (Future)
quantum_nerf = FullQuantumNeRF()  # To be implemented
```

---

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install torch torchvision matplotlib imageio configargparse tqdm
   ```

2. **Download Sample Data**
   ```bash
   # Download the lego dataset or use your own images
   ```

3. **Start Training**
   ```python
   # Run the training pipeline
   train_nerf()
   ```

4. **Render Novel Views**
   ```python
   # Generate new viewpoints after training
   render_novel_views(model, test_poses)
   ```

---

## Key Takeaways

- **Modular Design**: Each component can be independently upgraded
- **Quantum Ready**: Architecture designed for seamless quantum integration
- **Extensible**: Easy to add new features and optimizations
- **Educational**: Clear separation of concerns for learning

This implementation gives you a solid foundation to build upon and leaves clear pathways for quantum enhancements! 🚀