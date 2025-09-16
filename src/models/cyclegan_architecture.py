"""
CycleGAN Architecture for Cross-Genre Music Style Transfer
Implements generator and discriminator networks for multi-domain audio style transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

class ResidualBlock(nn.Module):
    """Residual block with instance normalization for generator bottleneck."""
    
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return residual + out

class Generator(nn.Module):
    """
    Generator network for mel-spectrogram style transfer.
    Architecture: Encoder -> Bottleneck (Residual blocks) -> Decoder
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        n_residual_blocks: int = 9,
        input_height: int = 128,
        input_width: int = 256
    ):
        super(Generator, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Encoder: Downsampling layers
        encoder = [
            # Initial convolution
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        in_channels = base_channels
        for i in range(2):  # 2 downsampling layers
            out_channels = in_channels * 2
            encoder += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder)
        
        # Bottleneck: Residual blocks
        bottleneck = []
        for _ in range(n_residual_blocks):
            bottleneck.append(ResidualBlock(in_channels))
        
        self.bottleneck = nn.Sequential(*bottleneck)
        
        # Decoder: Upsampling layers
        decoder = []
        for i in range(2):  # 2 upsampling layers
            out_channels = in_channels // 2
            decoder += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, 
                                 padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Final convolution
        decoder += [
            nn.Conv2d(base_channels, output_channels, kernel_size=7, padding=3),
            nn.Tanh()  # Output in [-1, 1] range
        ]
        
        self.decoder = nn.Sequential(*decoder)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.
        
        Args:
            x: Input mel-spectrogram [B, C, H, W]
            
        Returns:
            Generated mel-spectrogram [B, C, H, W]
        """
        encoded = self.encoder(x)
        bottleneck_out = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck_out)
        
        return decoded

class Discriminator(nn.Module):
    """
    PatchGAN discriminator for mel-spectrogram discrimination.
    Outputs a patch-based classification map instead of single value.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3
    ):
        super(Discriminator, self).__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        in_channels = base_channels
        for i in range(n_layers):
            out_channels = min(in_channels * 2, 512)  # Cap at 512 channels
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))
        # No activation - raw logits for loss computation
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input mel-spectrogram [B, C, H, W]
            
        Returns:
            Patch-based classification map [B, 1, H', W']
        """
        return self.model(x)

class CycleGAN(nn.Module):
    """
    Complete CycleGAN model for cross-genre music style transfer.
    Implements bidirectional generators and domain-specific discriminators.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_generator_channels: int = 64,
        base_discriminator_channels: int = 64,
        n_residual_blocks: int = 9,
        n_discriminator_layers: int = 3,
        input_height: int = 128,
        input_width: int = 256
    ):
        super(CycleGAN, self).__init__()
        
        # Generators for bidirectional translation
        # G_AB: Domain A (e.g., Folk) -> Domain B (e.g., Rock/Jazz)
        self.G_AB = Generator(
            input_channels=input_channels,
            output_channels=input_channels,
            base_channels=base_generator_channels,
            n_residual_blocks=n_residual_blocks,
            input_height=input_height,
            input_width=input_width
        )
        
        # G_BA: Domain B -> Domain A
        self.G_BA = Generator(
            input_channels=input_channels,
            output_channels=input_channels,
            base_channels=base_generator_channels,
            n_residual_blocks=n_residual_blocks,
            input_height=input_height,
            input_width=input_width
        )
        
        # Discriminators for each domain
        # D_A: Discriminates domain A samples
        self.D_A = Discriminator(
            input_channels=input_channels,
            base_channels=base_discriminator_channels,
            n_layers=n_discriminator_layers
        )
        
        # D_B: Discriminates domain B samples
        self.D_B = Discriminator(
            input_channels=input_channels,
            base_channels=base_discriminator_channels,
            n_layers=n_discriminator_layers
        )
    
    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor) -> dict:
        """
        Forward pass through CycleGAN.
        
        Args:
            x_A: Samples from domain A [B, C, H, W]
            x_B: Samples from domain B [B, C, H, W]
            
        Returns:
            Dictionary containing all generated samples and discriminator outputs
        """
        # Generate samples
        fake_B = self.G_AB(x_A)  # A -> B
        fake_A = self.G_BA(x_B)  # B -> A
        
        # Cycle consistency
        cycle_A = self.G_BA(fake_B)  # A -> B -> A
        cycle_B = self.G_AB(fake_A)  # B -> A -> B
        
        # Identity mapping (when input domain matches target)
        identity_A = self.G_BA(x_A)  # A -> A (should be identity)
        identity_B = self.G_AB(x_B)  # B -> B (should be identity)
        
        # Discriminator outputs
        D_A_real = self.D_A(x_A)
        D_A_fake = self.D_A(fake_A)
        
        D_B_real = self.D_B(x_B)
        D_B_fake = self.D_B(fake_B)
        
        return {
            # Generated samples
            'fake_A': fake_A,
            'fake_B': fake_B,
            
            # Cycle consistency
            'cycle_A': cycle_A,
            'cycle_B': cycle_B,
            
            # Identity mapping
            'identity_A': identity_A,
            'identity_B': identity_B,
            
            # Discriminator outputs
            'D_A_real': D_A_real,
            'D_A_fake': D_A_fake,
            'D_B_real': D_B_real,
            'D_B_fake': D_B_fake,
            
            # Real samples (for loss computation)
            'real_A': x_A,
            'real_B': x_B
        }
    
    def generate_A_to_B(self, x_A: torch.Tensor) -> torch.Tensor:
        """Generate domain B samples from domain A."""
        return self.G_AB(x_A)
    
    def generate_B_to_A(self, x_B: torch.Tensor) -> torch.Tensor:
        """Generate domain A samples from domain B."""
        return self.G_BA(x_B)

class StarGAN_VC(nn.Module):
    """
    Alternative StarGAN-VC architecture for multi-domain style transfer.
    Single generator with domain conditioning.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        n_domains: int = 3,  # Folk, Jazz, Rock
        base_channels: int = 64,
        n_residual_blocks: int = 6
    ):
        super(StarGAN_VC, self).__init__()
        
        self.n_domains = n_domains
        
        # Single generator with domain conditioning
        # Domain embedding
        self.domain_embedding = nn.Embedding(n_domains, 64)
        
        # Generator with conditional input
        self.generator = Generator(
            input_channels=input_channels + 1,  # +1 for domain conditioning
            output_channels=input_channels,
            base_channels=base_channels,
            n_residual_blocks=n_residual_blocks
        )
        
        # Single discriminator with domain classification
        self.discriminator = nn.Sequential(
            # Feature extraction
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*8),
            nn.LeakyReLU(0.2),
        )
        
        # Real/fake classification head
        self.adv_head = nn.Conv2d(base_channels*8, 1, 3, 1, 1)
        
        # Domain classification head
        self.domain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, n_domains)
        )
    
    def forward(self, x: torch.Tensor, target_domain: torch.Tensor) -> dict:
        """
        Forward pass through StarGAN-VC.
        
        Args:
            x: Input mel-spectrogram [B, C, H, W]
            target_domain: Target domain labels [B]
            
        Returns:
            Generated samples and discriminator outputs
        """
        # Generate domain conditioning map
        domain_emb = self.domain_embedding(target_domain)  # [B, 64]
        domain_map = domain_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        domain_map = domain_map.expand(-1, -1, x.size(2), x.size(3))  # [B, 64, H, W]
        
        # Concatenate with input
        conditioned_input = torch.cat([x, domain_map[:, :1]], dim=1)  # Use only first channel for conditioning
        
        # Generate
        generated = self.generator(conditioned_input)
        
        # Discriminate
        disc_features = self.discriminator(generated)
        adv_output = self.adv_head(disc_features)
        domain_output = self.domain_head(disc_features)
        
        return {
            'generated': generated,
            'adv_output': adv_output,
            'domain_output': domain_output
        }

def test_architectures():
    """Test the implemented architectures with sample data."""
    print("Testing CycleGAN and StarGAN-VC architectures...")
    
    # Test parameters
    batch_size = 4
    channels = 1
    height = 128
    width = 256
    
    # Create sample data
    x_A = torch.randn(batch_size, channels, height, width)
    x_B = torch.randn(batch_size, channels, height, width)
    
    # Test CycleGAN
    print("\n1. Testing CycleGAN...")
    cyclegan = CycleGAN(
        input_channels=channels,
        input_height=height,
        input_width=width
    )
    
    with torch.no_grad():
        output = cyclegan(x_A, x_B)
    
    print(f"   ✓ Input A shape: {x_A.shape}")
    print(f"   ✓ Input B shape: {x_B.shape}")
    print(f"   ✓ Generated A->B shape: {output['fake_B'].shape}")
    print(f"   ✓ Generated B->A shape: {output['fake_A'].shape}")
    print(f"   ✓ Cycle A shape: {output['cycle_A'].shape}")
    print(f"   ✓ Cycle B shape: {output['cycle_B'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in cyclegan.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    # Test StarGAN-VC
    print("\n2. Testing StarGAN-VC...")
    stargan = StarGAN_VC(
        input_channels=channels,
        n_domains=3
    )
    
    target_domains = torch.randint(0, 3, (batch_size,))
    
    with torch.no_grad():
        output = stargan(x_A, target_domains)
    
    print(f"   ✓ Input shape: {x_A.shape}")
    print(f"   ✓ Target domains: {target_domains}")
    print(f"   ✓ Generated shape: {output['generated'].shape}")
    print(f"   ✓ Adversarial output shape: {output['adv_output'].shape}")
    print(f"   ✓ Domain output shape: {output['domain_output'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in stargan.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    print("\n✓ All architecture tests passed!")

if __name__ == "__main__":
    test_architectures()