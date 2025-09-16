"""
Loss Functions for Cross-Genre Music Style Transfer
Implements adversarial, cycle consistency, identity, perceptual, and rhythm preservation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Dict, Tuple, Optional

class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    Supports both LSGAN and vanilla GAN losses.
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        prediction: torch.Tensor,
        is_real: bool,
        is_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Compute adversarial loss.
        
        Args:
            prediction: Discriminator output
            is_real: Whether the sample is real or fake
            is_discriminator: Whether computing loss for discriminator or generator
            
        Returns:
            Adversarial loss value
        """
        if self.loss_type == 'lsgan':
            if is_real:
                target = torch.ones_like(prediction)
            else:
                target = torch.zeros_like(prediction)
            loss = self.criterion(prediction, target)
        
        elif self.loss_type == 'vanilla':
            if is_discriminator:
                if is_real:
                    loss = self.criterion(prediction, torch.ones_like(prediction))
                else:
                    loss = self.criterion(prediction, torch.zeros_like(prediction))
            else:  # Generator loss
                loss = self.criterion(prediction, torch.ones_like(prediction))
        
        return loss

class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss to preserve content during style transfer.
    ||G_BA(G_AB(x)) - x||₁
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super(CycleConsistencyLoss, self).__init__()
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, cycle_output: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
        """
        Compute cycle consistency loss.
        
        Args:
            cycle_output: Output after cycle (A->B->A or B->A->B)
            original_input: Original input
            
        Returns:
            Cycle consistency loss
        """
        return self.criterion(cycle_output, original_input)

class IdentityLoss(nn.Module):
    """
    Identity loss to preserve input when target domain matches source domain.
    Helps with color consistency and reduces unnecessary changes.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super(IdentityLoss, self).__init__()
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, identity_output: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss.
        
        Args:
            identity_output: Output when input domain == target domain
            original_input: Original input
            
        Returns:
            Identity loss
        """
        return self.criterion(identity_output, original_input)

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Adapted for audio spectrograms by treating them as grayscale images.
    """
    
    def __init__(self, feature_layers: list = None, use_gpu: bool = True):
        super(PerceptualLoss, self).__init__()
        
        if feature_layers is None:
            feature_layers = ['conv_4']  # Use VGG conv4 features
        
        self.feature_layers = feature_layers
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Load pre-trained VGG network
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            
            if self.use_gpu:
                vgg = vgg.cuda()
            
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            
            self.vgg = vgg
            self.available = True
            
        except ImportError:
            print("Warning: torchvision not available. Perceptual loss will be disabled.")
            self.available = False
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using VGG features.
        
        Args:
            generated: Generated mel-spectrogram
            target: Target mel-spectrogram
            
        Returns:
            Perceptual loss
        """
        if not self.available:
            return torch.tensor(0.0, device=generated.device, requires_grad=True)
        
        # Convert single channel to RGB for VGG
        if generated.size(1) == 1:
            generated = generated.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to [0, 1] range
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        # Extract features
        gen_features = self._extract_features(generated)
        target_features = self._extract_features(target)
        
        # Compute L2 loss between features
        loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += F.mse_loss(gen_feat, target_feat)
        
        return loss / len(gen_features)
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract VGG features from specified layers."""
        features = []
        current_features = x
        
        layer_names = ['conv_1_1', 'conv_1_2', 'pool_1',
                      'conv_2_1', 'conv_2_2', 'pool_2',
                      'conv_3_1', 'conv_3_2', 'conv_3_3', 'conv_3_4', 'pool_3',
                      'conv_4_1', 'conv_4_2', 'conv_4_3', 'conv_4_4', 'pool_4']
        
        for i, layer in enumerate(self.vgg[:16]):  # Only use first few layers
            current_features = layer(current_features)
            layer_name = layer_names[min(i, len(layer_names)-1)]
            
            if any(target_layer in layer_name for target_layer in self.feature_layers):
                features.append(current_features)
        
        return features

class RhythmPreservationLoss(nn.Module):
    """
    Custom loss to preserve rhythmic characteristics during style transfer.
    Compares rhythm-related features between original and generated spectrograms.
    """
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        super(RhythmPreservationLoss, self).__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.criterion = nn.L1Loss()
    
    def forward(self, generated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Compute rhythm preservation loss.
        
        Args:
            generated: Generated mel-spectrogram [B, C, H, W]
            original: Original mel-spectrogram [B, C, H, W]
            
        Returns:
            Rhythm preservation loss
        """
        batch_size = generated.size(0)
        device = generated.device
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for i in range(batch_size):
            try:
                # Extract single samples and convert to numpy
                gen_spec = generated[i, 0].detach().cpu().numpy()
                orig_spec = original[i, 0].detach().cpu().numpy()
                
                # Compute rhythmic features
                gen_rhythm = self._extract_rhythm_features(gen_spec)
                orig_rhythm = self._extract_rhythm_features(orig_spec)
                
                # Convert back to tensors
                gen_rhythm_tensor = torch.tensor(gen_rhythm, device=device, dtype=torch.float32)
                orig_rhythm_tensor = torch.tensor(orig_rhythm, device=device, dtype=torch.float32)
                
                # Compute loss
                rhythm_loss = self.criterion(gen_rhythm_tensor, orig_rhythm_tensor)
                total_loss = total_loss + rhythm_loss
                
            except Exception as e:
                # If rhythm extraction fails, skip this sample
                continue
        
        return total_loss / batch_size
    
    def _extract_rhythm_features(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Extract rhythm-related features from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram as numpy array
            
        Returns:
            Rhythm feature vector
        """
        try:
            # Convert mel-spectrogram back to linear scale for rhythm analysis
            # This is a simplified approach - in practice, you'd want more sophisticated rhythm extraction
            
            # Compute onset strength
            onset_strength = librosa.onset.onset_strength(
                S=librosa.db_to_power(mel_spec),
                sr=self.sr,
                hop_length=self.hop_length
            )
            
            # Extract basic rhythm statistics
            rhythm_features = [
                np.mean(onset_strength),
                np.std(onset_strength),
                np.max(onset_strength),
                np.sum(onset_strength > np.mean(onset_strength) + np.std(onset_strength))  # Peak count
            ]
            
            return np.array(rhythm_features)
            
        except Exception:
            # Return zeros if extraction fails
            return np.zeros(4)

class SpectralLoss(nn.Module):
    """
    Spectral loss to preserve important frequency characteristics.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super(SpectralLoss, self).__init__()
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss focusing on frequency domain characteristics.
        
        Args:
            generated: Generated mel-spectrogram
            target: Target mel-spectrogram
            
        Returns:
            Spectral loss
        """
        # Compute spectral features
        gen_spectral = self._compute_spectral_features(generated)
        target_spectral = self._compute_spectral_features(target)
        
        return self.criterion(gen_spectral, target_spectral)
    
    def _compute_spectral_features(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral characteristics from mel-spectrogram."""
        # Compute spectral centroid, rolloff, etc. using torch operations
        
        # Spectral centroid (frequency-weighted average)
        freq_bins = torch.arange(mel_spec.size(2), device=mel_spec.device, dtype=torch.float32)
        freq_bins = freq_bins.view(1, 1, -1, 1)
        
        # Normalize and compute weighted average
        normalized_spec = F.softmax(mel_spec, dim=2)
        spectral_centroid = torch.sum(normalized_spec * freq_bins, dim=2, keepdim=True)
        
        # Spectral rolloff (cumulative energy)
        cumsum_spec = torch.cumsum(normalized_spec, dim=2)
        rolloff_threshold = 0.85
        rolloff_mask = cumsum_spec >= rolloff_threshold
        spectral_rolloff = torch.argmax(rolloff_mask.float(), dim=2, keepdim=True).float()
        
        # Combine features
        spectral_features = torch.cat([spectral_centroid, spectral_rolloff], dim=2)
        
        return spectral_features

class CombinedLoss(nn.Module):
    """
    Combined loss function for CycleGAN training.
    Combines all loss components with appropriate weights.
    """
    
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lambda_perceptual: float = 1.0,
        lambda_rhythm: float = 2.0,
        lambda_spectral: float = 1.0,
        adversarial_loss_type: str = 'lsgan'
    ):
        super(CombinedLoss, self).__init__()
        
        # Loss weights
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_perceptual = lambda_perceptual
        self.lambda_rhythm = lambda_rhythm
        self.lambda_spectral = lambda_spectral
        
        # Loss functions
        self.adversarial_loss = AdversarialLoss(adversarial_loss_type)
        self.cycle_loss = CycleConsistencyLoss()
        self.identity_loss = IdentityLoss()
        self.perceptual_loss = PerceptualLoss()
        self.rhythm_loss = RhythmPreservationLoss()
        self.spectral_loss = SpectralLoss()
    
    def compute_generator_loss(self, cyclegan_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute complete generator loss.
        
        Args:
            cyclegan_output: Output from CycleGAN forward pass
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # Adversarial losses
        adv_loss_A = self.adversarial_loss(cyclegan_output['D_A_fake'], is_real=True, is_discriminator=False)
        adv_loss_B = self.adversarial_loss(cyclegan_output['D_B_fake'], is_real=True, is_discriminator=False)
        losses['adversarial'] = adv_loss_A + adv_loss_B
        
        # Cycle consistency losses
        cycle_loss_A = self.cycle_loss(cyclegan_output['cycle_A'], cyclegan_output['real_A'])
        cycle_loss_B = self.cycle_loss(cyclegan_output['cycle_B'], cyclegan_output['real_B'])
        losses['cycle'] = self.lambda_cycle * (cycle_loss_A + cycle_loss_B)
        
        # Identity losses
        identity_loss_A = self.identity_loss(cyclegan_output['identity_A'], cyclegan_output['real_A'])
        identity_loss_B = self.identity_loss(cyclegan_output['identity_B'], cyclegan_output['real_B'])
        losses['identity'] = self.lambda_identity * (identity_loss_A + identity_loss_B)
        
        # Perceptual losses
        perceptual_loss_A = self.perceptual_loss(cyclegan_output['fake_A'], cyclegan_output['real_A'])
        perceptual_loss_B = self.perceptual_loss(cyclegan_output['fake_B'], cyclegan_output['real_B'])
        losses['perceptual'] = self.lambda_perceptual * (perceptual_loss_A + perceptual_loss_B)
        
        # Rhythm preservation losses
        rhythm_loss_A = self.rhythm_loss(cyclegan_output['fake_A'], cyclegan_output['real_A'])
        rhythm_loss_B = self.rhythm_loss(cyclegan_output['fake_B'], cyclegan_output['real_B'])
        losses['rhythm'] = self.lambda_rhythm * (rhythm_loss_A + rhythm_loss_B)
        
        # Spectral losses
        spectral_loss_A = self.spectral_loss(cyclegan_output['fake_A'], cyclegan_output['real_A'])
        spectral_loss_B = self.spectral_loss(cyclegan_output['fake_B'], cyclegan_output['real_B'])
        losses['spectral'] = self.lambda_spectral * (spectral_loss_A + spectral_loss_B)
        
        # Total generator loss
        losses['total'] = (losses['adversarial'] + losses['cycle'] + losses['identity'] + 
                          losses['perceptual'] + losses['rhythm'] + losses['spectral'])
        
        return losses
    
    def compute_discriminator_loss(self, cyclegan_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator losses.
        
        Args:
            cyclegan_output: Output from CycleGAN forward pass
            
        Returns:
            Dictionary of discriminator loss components
        """
        losses = {}
        
        # Discriminator A loss
        real_loss_A = self.adversarial_loss(cyclegan_output['D_A_real'], is_real=True, is_discriminator=True)
        fake_loss_A = self.adversarial_loss(cyclegan_output['D_A_fake'], is_real=False, is_discriminator=True)
        losses['D_A'] = (real_loss_A + fake_loss_A) * 0.5
        
        # Discriminator B loss
        real_loss_B = self.adversarial_loss(cyclegan_output['D_B_real'], is_real=True, is_discriminator=True)
        fake_loss_B = self.adversarial_loss(cyclegan_output['D_B_fake'], is_real=False, is_discriminator=True)
        losses['D_B'] = (real_loss_B + fake_loss_B) * 0.5
        
        # Total discriminator loss
        losses['total'] = losses['D_A'] + losses['D_B']
        
        return losses

def test_loss_functions():
    """Test all loss functions with sample data."""
    print("Testing loss functions...")
    
    # Create sample data
    batch_size = 2
    channels = 1
    height = 128
    width = 256
    
    real_A = torch.randn(batch_size, channels, height, width)
    real_B = torch.randn(batch_size, channels, height, width)
    fake_A = torch.randn(batch_size, channels, height, width)
    fake_B = torch.randn(batch_size, channels, height, width)
    
    # Test individual loss functions
    print("\n1. Testing individual loss functions...")
    
    # Adversarial loss
    adv_loss = AdversarialLoss()
    disc_output = torch.randn(batch_size, 1, 16, 32)
    loss_val = adv_loss(disc_output, is_real=True)
    print(f"   ✓ Adversarial loss: {loss_val.item():.4f}")
    
    # Cycle consistency loss
    cycle_loss = CycleConsistencyLoss()
    loss_val = cycle_loss(fake_A, real_A)
    print(f"   ✓ Cycle consistency loss: {loss_val.item():.4f}")
    
    # Identity loss
    identity_loss = IdentityLoss()
    loss_val = identity_loss(fake_A, real_A)
    print(f"   ✓ Identity loss: {loss_val.item():.4f}")
    
    # Perceptual loss
    perceptual_loss = PerceptualLoss()
    loss_val = perceptual_loss(fake_A, real_A)
    print(f"   ✓ Perceptual loss: {loss_val.item():.4f}")
    
    # Rhythm preservation loss
    rhythm_loss = RhythmPreservationLoss()
    loss_val = rhythm_loss(fake_A, real_A)
    print(f"   ✓ Rhythm preservation loss: {loss_val.item():.4f}")
    
    # Test combined loss
    print("\n2. Testing combined loss...")
    combined_loss = CombinedLoss()
    
    # Create mock CycleGAN output
    cyclegan_output = {
        'real_A': real_A,
        'real_B': real_B,
        'fake_A': fake_A,
        'fake_B': fake_B,
        'cycle_A': torch.randn_like(real_A),
        'cycle_B': torch.randn_like(real_B),
        'identity_A': torch.randn_like(real_A),
        'identity_B': torch.randn_like(real_B),
        'D_A_real': torch.randn(batch_size, 1, 16, 32),
        'D_A_fake': torch.randn(batch_size, 1, 16, 32),
        'D_B_real': torch.randn(batch_size, 1, 16, 32),
        'D_B_fake': torch.randn(batch_size, 1, 16, 32)
    }
    
    # Test generator loss
    gen_losses = combined_loss.compute_generator_loss(cyclegan_output)
    print(f"   ✓ Generator total loss: {gen_losses['total'].item():.4f}")
    print(f"     - Adversarial: {gen_losses['adversarial'].item():.4f}")
    print(f"     - Cycle: {gen_losses['cycle'].item():.4f}")
    print(f"     - Identity: {gen_losses['identity'].item():.4f}")
    print(f"     - Perceptual: {gen_losses['perceptual'].item():.4f}")
    print(f"     - Rhythm: {gen_losses['rhythm'].item():.4f}")
    print(f"     - Spectral: {gen_losses['spectral'].item():.4f}")
    
    # Test discriminator loss
    disc_losses = combined_loss.compute_discriminator_loss(cyclegan_output)
    print(f"   ✓ Discriminator total loss: {disc_losses['total'].item():.4f}")
    print(f"     - D_A: {disc_losses['D_A'].item():.4f}")
    print(f"     - D_B: {disc_losses['D_B'].item():.4f}")
    
    print("\n✓ All loss function tests passed!")

if __name__ == "__main__":
    test_loss_functions()