"""
Phase 4.4: Rhythm-Aware Loss Functions
=====================================

This module implements specialized loss functions that enforce rhythmic consistency
and preserve traditional folk rhythmic patterns during cross-genre style transfer.

Features:
- Beat-aligned loss functions for rhythmic consistency
- Tempo preservation losses
- Traditional pattern preservation constraints
- Rhythmic perceptual losses
- Multi-scale temporal coherence losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class BeatAlignmentLoss(nn.Module):
    """
    Loss function that enforces alignment between generated and target audio
    at beat positions for rhythmic consistency.
    """
    
    def __init__(self, sample_rate: int = 16000, weight: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.weight = weight
        
    def extract_beat_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract beat-aligned features from audio.
        
        Args:
            audio: Audio tensor [batch, channels, time] or [batch, time]
            
        Returns:
            Beat-aligned features
        """
        if len(audio.shape) == 3:
            # Multi-channel audio, take first channel
            audio_mono = audio[:, 0, :]
        else:
            audio_mono = audio
        
        batch_size = audio_mono.shape[0]
        beat_features = []
        
        for i in range(batch_size):
            audio_np = audio_mono[i].detach().cpu().numpy()
            
            # Extract onset strength (proxy for beat information)
            try:
                onset_strength = librosa.onset.onset_strength(
                    y=audio_np, sr=self.sample_rate, hop_length=512
                )
                
                # Pad or truncate to fixed length
                target_length = 128  # Fixed length for consistent processing
                if len(onset_strength) > target_length:
                    onset_strength = onset_strength[:target_length]
                else:
                    onset_strength = np.pad(onset_strength, 
                                          (0, target_length - len(onset_strength)))
                
                beat_features.append(onset_strength)
                
            except Exception:
                # Fallback: create zero features
                beat_features.append(np.zeros(128))
        
        return torch.FloatTensor(np.array(beat_features)).to(audio.device)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute beat alignment loss.
        
        Args:
            generated: Generated audio [batch, time] or [batch, channels, time]
            target: Target audio [batch, time] or [batch, channels, time]
            
        Returns:
            Beat alignment loss
        """
        # Extract beat features
        gen_beats = self.extract_beat_features(generated)
        target_beats = self.extract_beat_features(target)
        
        # Compute MSE loss between beat features
        beat_loss = F.mse_loss(gen_beats, target_beats)
        
        return self.weight * beat_loss

class TempoConsistencyLoss(nn.Module):
    """
    Loss function that enforces tempo consistency during style transfer.
    """
    
    def __init__(self, sample_rate: int = 16000, weight: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.weight = weight
        
    def estimate_tempo_tensor(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Estimate tempo from audio tensor using autocorrelation.
        
        Args:
            audio: Audio tensor [batch, time]
            
        Returns:
            Estimated tempo for each batch item
        """
        batch_size = audio.shape[0]
        tempos = []
        
        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()
            
            try:
                # Extract onset strength
                onset_strength = librosa.onset.onset_strength(
                    y=audio_np, sr=self.sample_rate, hop_length=512
                )
                
                # Compute autocorrelation
                autocorr = np.correlate(onset_strength, onset_strength, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Convert to tempo
                hop_length = 512
                time_axis = librosa.frames_to_time(
                    np.arange(len(autocorr)), sr=self.sample_rate, hop_length=hop_length
                )
                
                # Look for peak in reasonable tempo range
                min_period = 60 / 180  # 180 BPM
                max_period = 60 / 60   # 60 BPM
                
                valid_indices = np.where((time_axis >= min_period) & (time_axis <= max_period))[0]
                
                if len(valid_indices) > 0:
                    peak_idx = np.argmax(autocorr[valid_indices])
                    beat_period = time_axis[valid_indices[peak_idx]]
                    tempo = 60.0 / beat_period
                else:
                    tempo = 120.0  # Default
                
                tempos.append(tempo)
                
            except Exception:
                tempos.append(120.0)  # Default tempo
        
        return torch.FloatTensor(tempos).to(audio.device)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute tempo consistency loss.
        
        Args:
            generated: Generated audio [batch, time]
            target: Target audio [batch, time]
            
        Returns:
            Tempo consistency loss
        """
        if len(generated.shape) == 3:
            generated = generated[:, 0, :]  # Take first channel
        if len(target.shape) == 3:
            target = target[:, 0, :]
        
        # Estimate tempos
        gen_tempos = self.estimate_tempo_tensor(generated)
        target_tempos = self.estimate_tempo_tensor(target)
        
        # Compute relative tempo difference
        tempo_ratio = gen_tempos / (target_tempos + 1e-8)
        
        # Loss is deviation from 1.0 (same tempo)
        tempo_loss = F.mse_loss(tempo_ratio, torch.ones_like(tempo_ratio))
        
        return self.weight * tempo_loss

class RhythmicPatternLoss(nn.Module):
    """
    Loss function that preserves rhythmic patterns during style transfer.
    """
    
    def __init__(self, sample_rate: int = 16000, pattern_length: int = 16, weight: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.pattern_length = pattern_length  # 16th note resolution
        self.weight = weight
        
    def extract_rhythmic_pattern(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract rhythmic pattern representation.
        
        Args:
            audio: Audio tensor [batch, time]
            
        Returns:
            Rhythmic pattern tensor [batch, pattern_length]
        """
        batch_size = audio.shape[0]
        patterns = []
        
        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()
            
            try:
                # Extract onset strength
                onset_strength = librosa.onset.onset_strength(
                    y=audio_np, sr=self.sample_rate, hop_length=512
                )
                
                # Simple beat tracking
                tempo, beats = librosa.beat.beat_track(
                    onset_envelope=onset_strength, sr=self.sample_rate, hop_length=512
                )
                
                if len(beats) > 1:
                    # Extract pattern for first beat interval
                    beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=512)
                    
                    # Use first beat interval as reference
                    beat_duration = beat_times[1] - beat_times[0]
                    
                    # Create pattern representation
                    pattern = np.zeros(self.pattern_length)
                    
                    # Map onsets to pattern positions
                    onset_times = librosa.frames_to_time(
                        np.arange(len(onset_strength)), sr=self.sample_rate, hop_length=512
                    )
                    
                    # Consider onsets in first beat
                    beat_start = beat_times[0]
                    beat_end = beat_times[1] if len(beat_times) > 1 else beat_start + beat_duration
                    
                    for j, (onset_time, strength) in enumerate(zip(onset_times, onset_strength)):
                        if beat_start <= onset_time < beat_end:
                            # Map to pattern position
                            relative_pos = (onset_time - beat_start) / beat_duration
                            pattern_idx = int(relative_pos * self.pattern_length)
                            if 0 <= pattern_idx < self.pattern_length:
                                pattern[pattern_idx] = max(pattern[pattern_idx], strength)
                
                else:
                    # No beats found, create default pattern
                    pattern = np.zeros(self.pattern_length)
                    pattern[0] = 1.0  # Strong downbeat
                
                patterns.append(pattern)
                
            except Exception:
                # Default pattern on failure
                pattern = np.zeros(self.pattern_length)
                pattern[0] = 1.0
                patterns.append(pattern)
        
        return torch.FloatTensor(np.array(patterns)).to(audio.device)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute rhythmic pattern preservation loss.
        
        Args:
            generated: Generated audio [batch, time]
            target: Target audio [batch, time]
            
        Returns:
            Rhythmic pattern loss
        """
        if len(generated.shape) == 3:
            generated = generated[:, 0, :]
        if len(target.shape) == 3:
            target = target[:, 0, :]
        
        # Extract rhythmic patterns
        gen_patterns = self.extract_rhythmic_pattern(generated)
        target_patterns = self.extract_rhythmic_pattern(target)
        
        # Normalize patterns
        gen_patterns_norm = F.normalize(gen_patterns, p=2, dim=1)
        target_patterns_norm = F.normalize(target_patterns, p=2, dim=1)
        
        # Compute cosine similarity loss
        similarity = F.cosine_similarity(gen_patterns_norm, target_patterns_norm, dim=1)
        pattern_loss = 1.0 - similarity.mean()  # Convert similarity to loss
        
        return self.weight * pattern_loss

class PerceptualRhythmLoss(nn.Module):
    """
    Perceptual loss function for rhythmic content using learned features.
    """
    
    def __init__(self, sample_rate: int = 16000, weight: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.weight = weight
        
        # Simple CNN for rhythmic feature extraction
        self.feature_extractor = nn.Sequential(
            # Input: [batch, 1, n_mels, time]
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def audio_to_melspec(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel spectrogram.
        
        Args:
            audio: Audio tensor [batch, time]
            
        Returns:
            Mel spectrogram [batch, 1, n_mels, time]
        """
        batch_size = audio.shape[0]
        mel_specs = []
        
        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np, sr=self.sample_rate, n_mels=64, hop_length=512
            )
            
            # Convert to log scale
            mel_db = librosa.power_to_db(mel_spec)
            
            # Normalize
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
            
            # Ensure fixed size
            target_time = 128
            if mel_norm.shape[1] > target_time:
                mel_norm = mel_norm[:, :target_time]
            else:
                mel_norm = np.pad(mel_norm, ((0, 0), (0, target_time - mel_norm.shape[1])))
            
            mel_specs.append(mel_norm)
        
        # Convert to tensor and add channel dimension
        mel_tensor = torch.FloatTensor(np.array(mel_specs)).unsqueeze(1)
        return mel_tensor.to(audio.device)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual rhythm loss.
        
        Args:
            generated: Generated audio [batch, time]
            target: Target audio [batch, time]
            
        Returns:
            Perceptual rhythm loss
        """
        if len(generated.shape) == 3:
            generated = generated[:, 0, :]
        if len(target.shape) == 3:
            target = target[:, 0, :]
        
        # Convert to mel spectrograms
        gen_mel = self.audio_to_melspec(generated)
        target_mel = self.audio_to_melspec(target)
        
        # Extract features
        gen_features = self.feature_extractor(gen_mel)
        target_features = self.feature_extractor(target_mel)
        
        # Compute feature loss
        feature_loss = F.mse_loss(gen_features, target_features)
        
        return self.weight * feature_loss

class MultiScaleRhythmLoss(nn.Module):
    """
    Multi-scale rhythmic loss that considers rhythm at different temporal scales.
    """
    
    def __init__(self, sample_rate: int = 16000, scales: List[int] = [1, 2, 4, 8], weight: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.scales = scales
        self.weight = weight
        
    def compute_rhythmic_autocorr(self, audio: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Compute rhythmic autocorrelation at given scale.
        
        Args:
            audio: Audio tensor [batch, time]
            scale: Temporal scale factor
            
        Returns:
            Autocorrelation features
        """
        batch_size = audio.shape[0]
        autocorr_features = []
        
        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()
            
            # Downsample for different scales
            if scale > 1:
                # Simple downsampling
                audio_scaled = audio_np[::scale]
            else:
                audio_scaled = audio_np
            
            if len(audio_scaled) > 100:  # Minimum length check
                # Compute onset strength
                onset_strength = librosa.onset.onset_strength(
                    y=audio_scaled, sr=self.sample_rate // scale, hop_length=256
                )
                
                # Compute autocorrelation
                if len(onset_strength) > 10:
                    autocorr = np.correlate(onset_strength, onset_strength, mode='same')
                    
                    # Take central part and normalize
                    center = len(autocorr) // 2
                    half_window = min(32, center)  # Take at most 64 points
                    autocorr_window = autocorr[center-half_window:center+half_window]
                    
                    # Normalize
                    autocorr_norm = autocorr_window / (np.max(np.abs(autocorr_window)) + 1e-8)
                    
                    # Pad to fixed size
                    target_size = 64
                    if len(autocorr_norm) < target_size:
                        autocorr_norm = np.pad(autocorr_norm, 
                                             (0, target_size - len(autocorr_norm)))
                    else:
                        autocorr_norm = autocorr_norm[:target_size]
                    
                    autocorr_features.append(autocorr_norm)
                else:
                    autocorr_features.append(np.zeros(64))
            else:
                autocorr_features.append(np.zeros(64))
        
        return torch.FloatTensor(np.array(autocorr_features)).to(audio.device)
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale rhythm loss.
        
        Args:
            generated: Generated audio [batch, time]
            target: Target audio [batch, time]
            
        Returns:
            Multi-scale rhythm loss
        """
        if len(generated.shape) == 3:
            generated = generated[:, 0, :]
        if len(target.shape) == 3:
            target = target[:, 0, :]
        
        total_loss = 0.0
        
        for scale in self.scales:
            # Compute autocorrelation features at this scale
            gen_autocorr = self.compute_rhythmic_autocorr(generated, scale)
            target_autocorr = self.compute_rhythmic_autocorr(target, scale)
            
            # Compute loss at this scale
            scale_loss = F.mse_loss(gen_autocorr, target_autocorr)
            total_loss += scale_loss / len(self.scales)
        
        return self.weight * total_loss

class RhythmAwareLossCollection(nn.Module):
    """
    Collection of all rhythm-aware loss functions with configurable weights.
    """
    
    def __init__(self, sample_rate: int = 16000, config: Optional[Dict] = None):
        super().__init__()
        
        # Default configuration
        default_config = {
            'beat_alignment_weight': 1.0,
            'tempo_consistency_weight': 0.5,
            'rhythmic_pattern_weight': 1.0,
            'perceptual_rhythm_weight': 0.3,
            'multiscale_rhythm_weight': 0.7
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # Initialize loss functions
        self.beat_alignment_loss = BeatAlignmentLoss(
            sample_rate, weight=default_config['beat_alignment_weight']
        )
        
        self.tempo_consistency_loss = TempoConsistencyLoss(
            sample_rate, weight=default_config['tempo_consistency_weight']
        )
        
        self.rhythmic_pattern_loss = RhythmicPatternLoss(
            sample_rate, weight=default_config['rhythmic_pattern_weight']
        )
        
        self.perceptual_rhythm_loss = PerceptualRhythmLoss(
            sample_rate, weight=default_config['perceptual_rhythm_weight']
        )
        
        self.multiscale_rhythm_loss = MultiScaleRhythmLoss(
            sample_rate, weight=default_config['multiscale_rhythm_weight']
        )
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all rhythm-aware losses.
        
        Args:
            generated: Generated audio [batch, time] or [batch, channels, time]
            target: Target audio [batch, time] or [batch, channels, time]
            
        Returns:
            Dictionary of individual losses and total loss
        """
        losses = {}
        
        # Compute individual losses
        losses['beat_alignment'] = self.beat_alignment_loss(generated, target)
        losses['tempo_consistency'] = self.tempo_consistency_loss(generated, target)
        losses['rhythmic_pattern'] = self.rhythmic_pattern_loss(generated, target)
        losses['perceptual_rhythm'] = self.perceptual_rhythm_loss(generated, target)
        losses['multiscale_rhythm'] = self.multiscale_rhythm_loss(generated, target)
        
        # Compute total loss
        losses['total_rhythm_loss'] = sum(losses.values())
        
        return losses

def test_rhythm_aware_losses():
    """Test rhythm-aware loss functions."""
    print("Testing Rhythm-Aware Loss Functions")
    print("=" * 50)
    
    # Create test audio tensors
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    
    # Generate test audio with rhythmic patterns
    t = torch.linspace(0, 3, audio_length)
    
    # Source audio with one rhythmic pattern
    source_audio = torch.zeros(batch_size, audio_length)
    for i in range(batch_size):
        # Create periodic impulses (simulating beats)
        beat_freq = 2.0 + i * 0.5  # Different tempos for each batch item
        beats = torch.sin(2 * torch.pi * beat_freq * t)
        source_audio[i] = beats + 0.1 * torch.randn(audio_length)
    
    # Target audio with different rhythmic pattern
    target_audio = torch.zeros(batch_size, audio_length)
    for i in range(batch_size):
        beat_freq = 2.5 + i * 0.3  # Slightly different tempos
        beats = torch.sin(2 * torch.pi * beat_freq * t) + 0.5 * torch.sin(2 * torch.pi * beat_freq * 2 * t)
        target_audio[i] = beats + 0.1 * torch.randn(audio_length)
    
    # Test individual loss functions
    print("Testing individual loss functions...")
    
    # Beat Alignment Loss
    beat_loss_fn = BeatAlignmentLoss(sample_rate=16000, weight=1.0)
    beat_loss = beat_loss_fn(source_audio, target_audio)
    print(f"Beat Alignment Loss: {beat_loss.item():.4f}")
    
    # Tempo Consistency Loss
    tempo_loss_fn = TempoConsistencyLoss(sample_rate=16000, weight=1.0)
    tempo_loss = tempo_loss_fn(source_audio, target_audio)
    print(f"Tempo Consistency Loss: {tempo_loss.item():.4f}")
    
    # Rhythmic Pattern Loss
    pattern_loss_fn = RhythmicPatternLoss(sample_rate=16000, weight=1.0)
    pattern_loss = pattern_loss_fn(source_audio, target_audio)
    print(f"Rhythmic Pattern Loss: {pattern_loss.item():.4f}")
    
    # Perceptual Rhythm Loss
    perceptual_loss_fn = PerceptualRhythmLoss(sample_rate=16000, weight=1.0)
    perceptual_loss = perceptual_loss_fn(source_audio, target_audio)
    print(f"Perceptual Rhythm Loss: {perceptual_loss.item():.4f}")
    
    # Multi-Scale Rhythm Loss
    multiscale_loss_fn = MultiScaleRhythmLoss(sample_rate=16000, weight=1.0)
    multiscale_loss = multiscale_loss_fn(source_audio, target_audio)
    print(f"Multi-Scale Rhythm Loss: {multiscale_loss.item():.4f}")
    
    # Test complete loss collection
    print("\nTesting complete loss collection...")
    rhythm_losses = RhythmAwareLossCollection(sample_rate=16000)
    
    all_losses = rhythm_losses(source_audio, target_audio)
    
    print("All rhythm-aware losses:")
    for loss_name, loss_value in all_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Test with identical audio (should have low loss)
    print("\nTesting with identical audio (should have low losses)...")
    identical_losses = rhythm_losses(source_audio, source_audio)
    
    print("Identical audio losses:")
    for loss_name, loss_value in identical_losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    print("\nRhythm-aware loss function testing complete!")

if __name__ == "__main__":
    test_rhythm_aware_losses()