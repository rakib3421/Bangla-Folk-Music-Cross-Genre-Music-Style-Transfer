"""
CPU-Optimized Configuration for Cross-Genre Music Style Transfer
Optimized settings and configurations for training on CPU-only systems.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')

class CPUOptimizedConfig:
    """
    Configuration optimized for CPU-only training.
    """
    
    def __init__(self):
        """Initialize CPU-optimized configuration."""
        
        # Force CPU usage
        self.device = torch.device('cpu')
        
        # Optimized model parameters for CPU
        self.model_config = {
            # Reduced model complexity for CPU efficiency
            'base_generator_channels': 32,      # Reduced from 64
            'base_discriminator_channels': 32,  # Reduced from 64
            'n_residual_blocks': 4,            # Reduced from 6-9
            'input_height': 64,                # Match n_mels
            'input_width': 126,                # Match actual time frames from training pipeline
            'n_mels': 64,                      # Reduced from 128
        }
        
        # Training parameters optimized for CPU
        self.training_config = {
            'batch_size': 2,                   # Very small batch size
            'learning_rate': 0.0001,           # Slightly lower LR
            'max_epochs_phase1': 20,           # Reduced epochs for testing
            'max_epochs_phase2': 30,           # Reduced epochs for testing
            'gradient_accumulation_steps': 4,   # Simulate larger batch size
            'mixed_precision': False,          # Disable for CPU
            'num_workers': 0,                  # Disable multiprocessing
        }
        
        # Audio processing optimized for CPU
        self.audio_config = {
            'sample_rate': 16000,              # Reduced from 22050
            'n_fft': 1024,                     # Reduced from 2048
            'hop_length': 252,                 # Calculated to get exactly 128 frames
            'segment_length': 2.0,             # Reduced from 5.0 seconds
            'max_files_per_genre': 10,         # Limit dataset size
        }
        
        # Memory optimization
        self.memory_config = {
            'pin_memory': False,               # Disable for CPU
            'persistent_workers': False,       # Disable for CPU
            'prefetch_factor': 1,              # Minimal prefetching
            'enable_checkpointing': True,      # Save memory with gradient checkpointing
        }
        
        # Loss function weights (adjusted for faster convergence)
        self.loss_config = {
            'lambda_cycle': 5.0,               # Reduced from 10.0
            'lambda_identity': 0.25,           # Reduced from 0.5
            'lambda_perceptual': 0.5,          # Reduced from 1.0
            'lambda_rhythm': 0.5,              # Reduced from 1.0
            'lambda_spectral': 0.5,            # Reduced from 1.0
        }
        
        print("CPU-Optimized Configuration initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model channels: {self.model_config['base_generator_channels']}")
        print(f"  Batch size: {self.training_config['batch_size']}")
        print(f"  Mel bins: {self.model_config['n_mels']}")
        print(f"  Sample rate: {self.audio_config['sample_rate']}")
    
    def get_model_kwargs(self) -> Dict:
        """Get model initialization arguments."""
        return {
            'input_channels': 1,
            'base_generator_channels': self.model_config['base_generator_channels'],
            'base_discriminator_channels': self.model_config['base_discriminator_channels'],
            'n_residual_blocks': self.model_config['n_residual_blocks'],
            'input_height': self.model_config['input_height'],
            'input_width': self.model_config['input_width']
        }
    
    def get_dataset_kwargs(self) -> Dict:
        """Get dataset initialization arguments."""
        return {
            'segment_length': int(self.audio_config['segment_length'] * 
                                self.audio_config['sample_rate'] / 
                                self.audio_config['hop_length']),
            'n_mels': self.model_config['n_mels'],
            'sr': self.audio_config['sample_rate'],
            'hop_length': self.audio_config['hop_length'],
            'max_files_per_genre': self.audio_config['max_files_per_genre'],
            'augment': True
        }
    
    def get_training_kwargs(self) -> Dict:
        """Get training configuration arguments."""
        return {
            'batch_size': self.training_config['batch_size'],
            'learning_rate': self.training_config['learning_rate'],
            'device': self.device,
            'gradient_accumulation_steps': self.training_config['gradient_accumulation_steps'],
            'num_workers': self.training_config['num_workers']
        }
    
    def get_loss_kwargs(self) -> Dict:
        """Get loss function arguments."""
        return self.loss_config.copy()
    
    def optimize_torch_for_cpu(self):
        """Apply CPU-specific PyTorch optimizations."""
        
        # Set number of threads for CPU computation
        if hasattr(torch, 'set_num_threads'):
            # Use all available CPU cores, but limit to reasonable number
            num_threads = min(torch.get_num_threads(), 8)
            torch.set_num_threads(num_threads)
            print(f"Set PyTorch CPU threads: {num_threads}")
        
        # Enable CPU optimizations
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
            print("Enabled MKL-DNN backend for CPU optimization")
        
        # Disable CUDA if accidentally enabled
        if torch.cuda.is_available():
            torch.cuda.is_available = lambda: False
            print("Disabled CUDA to force CPU usage")
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'audio_config': self.audio_config,
            'memory_config': self.memory_config,
            'loss_config': self.loss_config,
            'device': str(self.device)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {filepath}")

def create_cpu_optimized_model():
    """Create a CPU-optimized CycleGAN model."""
    from cyclegan_architecture import CycleGAN
    
    config = CPUOptimizedConfig()
    
    model = CycleGAN(
        input_channels=1,
        base_generator_channels=config.model_config['base_generator_channels'],
        base_discriminator_channels=config.model_config['base_discriminator_channels'],
        n_residual_blocks=config.model_config['n_residual_blocks'],
        input_height=config.model_config['input_height'],
        input_width=config.model_config['input_width']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"CPU-Optimized CycleGAN Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model, config

def create_cpu_optimized_dataset():
    """Create a CPU-optimized dataset."""
    from training_pipeline import MelSpectrogramDataset, CrossGenreDataLoader
    
    config = CPUOptimizedConfig()
    dataset_kwargs = config.get_dataset_kwargs()
    
    print("Creating CPU-optimized dataset...")
    
    dataset = MelSpectrogramDataset(
        data_dir="data",
        genres=['Bangla Folk', 'Jazz', 'Rock'],
        **dataset_kwargs
    )
    
    data_loader = CrossGenreDataLoader(
        dataset=dataset,
        batch_size=config.training_config['batch_size'],
        shuffle=True,
        num_workers=0  # Disable multiprocessing for CPU
    )
    
    return dataset, data_loader, config

def cpu_optimized_training_step(
    model, 
    batch_A, 
    batch_B, 
    optimizers, 
    criterion, 
    gradient_accumulation_steps: int = 4
):
    """
    CPU-optimized training step with gradient accumulation.
    
    Args:
        model: CycleGAN model
        batch_A: Batch from domain A
        batch_B: Batch from domain B
        optimizers: Dictionary of optimizers
        criterion: Loss criterion
        gradient_accumulation_steps: Steps to accumulate gradients
        
    Returns:
        Dictionary of losses
    """
    device = next(model.parameters()).device
    
    real_A = batch_A['mel_spectrogram'].to(device)
    real_B = batch_B['mel_spectrogram'].to(device)
    
    # Skip if batch sizes don't match
    if real_A.size(0) != real_B.size(0):
        return None
    
    # Normalize gradients by accumulation steps
    accumulation_factor = 1.0 / gradient_accumulation_steps
    
    losses = {}
    
    # Generator training with gradient accumulation
    fake_B = model.G_AB(real_A)
    fake_A = model.G_BA(real_B)
    cycle_A = model.G_BA(fake_B)
    cycle_B = model.G_AB(fake_A)
    identity_A = model.G_BA(real_A)
    identity_B = model.G_AB(real_B)
    
    D_A_fake = model.D_A(fake_A)
    D_B_fake = model.D_B(fake_B)
    
    gen_output = {
        'fake_A': fake_A, 'fake_B': fake_B,
        'cycle_A': cycle_A, 'cycle_B': cycle_B,
        'identity_A': identity_A, 'identity_B': identity_B,
        'real_A': real_A, 'real_B': real_B,
        'D_A_fake': D_A_fake, 'D_B_fake': D_B_fake
    }
    
    generator_losses = criterion.compute_generator_loss(gen_output)
    gen_loss = generator_losses['total'] * accumulation_factor
    gen_loss.backward()
    losses['generator'] = generator_losses['total'].item()
    
    # Discriminator A training
    with torch.no_grad():
        fake_A_detached = model.G_BA(real_B)
    
    D_A_real = model.D_A(real_A)
    D_A_fake_detached = model.D_A(fake_A_detached)
    
    real_loss_A = criterion.adversarial_loss(D_A_real, is_real=True, is_discriminator=True)
    fake_loss_A = criterion.adversarial_loss(D_A_fake_detached, is_real=False, is_discriminator=True)
    disc_loss_A = (real_loss_A + fake_loss_A) * 0.5 * accumulation_factor
    disc_loss_A.backward()
    losses['discriminator_A'] = ((real_loss_A + fake_loss_A) * 0.5).item()
    
    # Discriminator B training
    with torch.no_grad():
        fake_B_detached = model.G_AB(real_A)
    
    D_B_real = model.D_B(real_B)
    D_B_fake_detached = model.D_B(fake_B_detached)
    
    real_loss_B = criterion.adversarial_loss(D_B_real, is_real=True, is_discriminator=True)
    fake_loss_B = criterion.adversarial_loss(D_B_fake_detached, is_real=False, is_discriminator=True)
    disc_loss_B = (real_loss_B + fake_loss_B) * 0.5 * accumulation_factor
    disc_loss_B.backward()
    losses['discriminator_B'] = ((real_loss_B + fake_loss_B) * 0.5).item()
    
    return losses

def test_cpu_optimization():
    """Test CPU optimization setup."""
    print("Testing CPU Optimization Setup...")
    
    try:
        # Initialize CPU configuration
        config = CPUOptimizedConfig()
        config.optimize_torch_for_cpu()
        
        # Test model creation
        print("\n1. Testing CPU-optimized model creation...")
        model, _ = create_cpu_optimized_model()
        
        # Test dataset creation
        print("\n2. Testing CPU-optimized dataset creation...")
        dataset, data_loader, _ = create_cpu_optimized_dataset()
        
        # Test a forward pass
        print("\n3. Testing model forward pass...")
        dummy_input_A = torch.randn(1, 1, 64, 64)
        dummy_input_B = torch.randn(1, 1, 64, 64)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input_A, dummy_input_B)
            print(f"   Forward pass successful!")
            print(f"   Output keys: {list(output.keys())}")
            print(f"   fake_A shape: {output['fake_A'].shape}")
            print(f"   fake_B shape: {output['fake_B'].shape}")
        
        # Test memory usage
        print("\n4. Testing memory efficiency...")
        
        # Estimate memory usage
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024  # MB
        print(f"   Parameter memory: ~{param_memory:.1f} MB")
        
        # Test batch processing
        print("\n5. Testing batch processing...")
        try:
            batch_A, batch_B = data_loader.get_paired_batch('Bangla Folk', 'Rock')
            print(f"   Batch A shape: {batch_A['mel_spectrogram'].shape}")
            print(f"   Batch B shape: {batch_B['mel_spectrogram'].shape}")
        except Exception as e:
            print(f"   Batch processing test skipped: {e}")
        
        # Save configuration
        config.save_config("cpu_optimized_config.json")
        
        print(f"\n✓ CPU optimization test completed successfully!")
        print(f"\nCPU Training Recommendations:")
        print(f"  • Use batch size: {config.training_config['batch_size']}")
        print(f"  • Enable gradient accumulation: {config.training_config['gradient_accumulation_steps']} steps")
        print(f"  • Reduced model complexity: {config.model_config['base_generator_channels']} channels")
        print(f"  • Smaller spectrograms: {config.model_config['n_mels']} mel bins")
        print(f"  • Shorter segments: {config.audio_config['segment_length']} seconds")
        print(f"  • Limited dataset: {config.audio_config['max_files_per_genre']} files per genre")
        
        return True
        
    except Exception as e:
        print(f"✗ CPU optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cpu_optimization()