"""
Phase 4.1: Source Separation for Vocal/Instrumental Isolation
==============================================================

This module implements multiple source separation techniques to isolate vocals
and instrumental components for targeted style transfer while preserving lyrics.

Features:
- Spleeter-based separation for high-quality vocal isolation
- PyTorch-based neural separation models
- CPU-optimized separation for resource-constrained systems
- Spectral masking techniques for basic separation
- Quality assessment and confidence scoring
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Try to import spleeter (may not be available)
try:
    from spleeter.separator import Separator
    SPLEETER_AVAILABLE = True
except ImportError:
    SPLEETER_AVAILABLE = False
    print("Spleeter not available. Using PyTorch-based separation only.")

class SpectralMaskingSeparator:
    """
    Basic spectral masking-based separation using harmonic/percussive separation
    and spectral filtering techniques. CPU-efficient fallback method.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def separate_harmonic_percussive(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate harmonic and percussive components using librosa.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (harmonic, percussive) components
        """
        # Compute STFT
        stft = librosa.stft(audio, hop_length=512, n_fft=2048)
        
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.decompose.hpss(stft, margin=3.0)
        
        # Convert back to time domain
        harmonic_audio = librosa.istft(harmonic, hop_length=512)
        percussive_audio = librosa.istft(percussive, hop_length=512)
        
        return harmonic_audio, percussive_audio
    
    def estimate_vocal_mask(self, audio: np.ndarray) -> np.ndarray:
        """
        Estimate vocal regions using spectral characteristics.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Vocal mask (1 = vocal, 0 = non-vocal)
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=128
        )
        mel_db = librosa.power_to_db(mel_spec)
        
        # Compute spectral centroid (brightness indicator)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )
        
        # Compute spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate
        )
        
        # Simple heuristic: vocals tend to have specific frequency characteristics
        # This is a basic approach - more sophisticated methods exist
        vocal_freq_range = (300, 3400)  # Hz, typical vocal range
        
        # Create frequency bins
        freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=self.sample_rate//2)
        vocal_bins = np.where((freqs >= vocal_freq_range[0]) & 
                              (freqs <= vocal_freq_range[1]))[0]
        
        # Estimate vocal presence based on energy in vocal frequency range
        vocal_energy = np.mean(mel_db[vocal_bins, :], axis=0)
        total_energy = np.mean(mel_db, axis=0)
        
        # Normalize and threshold
        vocal_ratio = vocal_energy / (total_energy + 1e-10)
        vocal_mask = (vocal_ratio > np.percentile(vocal_ratio, 60)).astype(float)
        
        return vocal_mask
    
    def separate_vocals_instruments(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate vocals and instruments using spectral masking.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (vocals, instruments) estimates
        """
        # First, separate harmonic and percussive
        harmonic, percussive = self.separate_harmonic_percussive(audio)
        
        # Estimate vocal mask
        vocal_mask = self.estimate_vocal_mask(audio)
        
        # Apply temporal smoothing to mask
        from scipy import ndimage
        vocal_mask_smooth = ndimage.gaussian_filter1d(vocal_mask, sigma=2.0)
        
        # Convert mask to audio length
        mask_interp = np.interp(
            np.linspace(0, len(vocal_mask_smooth)-1, len(audio)),
            np.arange(len(vocal_mask_smooth)),
            vocal_mask_smooth
        )
        
        # Apply mask to harmonic component (vocals are typically harmonic)
        vocal_estimate = harmonic * mask_interp
        instrument_estimate = audio - vocal_estimate
        
        # Ensure estimates sum to original (conservation of energy)
        total_estimate = vocal_estimate + instrument_estimate
        scale_factor = np.sqrt(np.mean(audio**2) / (np.mean(total_estimate**2) + 1e-10))
        
        vocal_estimate *= scale_factor
        instrument_estimate *= scale_factor
        
        return vocal_estimate, instrument_estimate

class PyTorchSeparationModel(nn.Module):
    """
    Lightweight PyTorch-based separation model for CPU training.
    Uses U-Net architecture for vocal/instrumental separation.
    """
    
    def __init__(self, n_fft: int = 2048, n_mels: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: [batch, 1, n_mels, time]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder for vocals
        self.vocal_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Mask output [0, 1]
        )
        
        # Decoder for instruments
        self.instrument_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Mask output [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass for separation.
        
        Args:
            x: Input mel spectrogram [batch, 1, n_mels, time]
            
        Returns:
            Tuple of (vocal_mask, instrument_mask)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode to masks
        vocal_mask = self.vocal_decoder(encoded)
        instrument_mask = self.instrument_decoder(encoded)
        
        # Ensure masks sum to approximately 1 (soft constraint)
        total_mask = vocal_mask + instrument_mask
        vocal_mask = vocal_mask / (total_mask + 1e-8)
        instrument_mask = instrument_mask / (total_mask + 1e-8)
        
        return vocal_mask, instrument_mask

class SpleeterSeparator:
    """
    Wrapper for Spleeter-based separation.
    """
    
    def __init__(self, model_name: str = '2stems-16kHz'):
        if not SPLEETER_AVAILABLE:
            raise ImportError("Spleeter not available. Install with: pip install spleeter")
        
        self.separator = Separator(model_name)
        self.model_name = model_name
        
    def separate(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        Separate audio using Spleeter.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with separated sources
        """
        # Ensure audio is 2D (stereo or mono converted to stereo)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=-1)
        
        # Spleeter expects specific sample rate
        if sample_rate != 16000:
            audio_resampled = librosa.resample(audio.T, orig_sr=sample_rate, target_sr=16000).T
        else:
            audio_resampled = audio
        
        # Separate
        sources = self.separator.separate(audio_resampled)
        
        # Convert back to original sample rate if needed
        if sample_rate != 16000:
            for key in sources:
                sources[key] = librosa.resample(
                    sources[key].T, orig_sr=16000, target_sr=sample_rate
                ).T
        
        # Convert to mono if needed
        for key in sources:
            if len(sources[key].shape) == 2:
                sources[key] = np.mean(sources[key], axis=-1)
        
        return sources

class SourceSeparationPipeline:
    """
    Complete source separation pipeline with multiple methods and quality assessment.
    """
    
    def __init__(self, sample_rate: int = 16000, method: str = 'auto'):
        """
        Initialize separation pipeline.
        
        Args:
            sample_rate: Target sample rate
            method: 'spleeter', 'pytorch', 'spectral', or 'auto'
        """
        self.sample_rate = sample_rate
        self.method = method
        
        # Initialize available separators
        self.spectral_separator = SpectralMaskingSeparator(sample_rate)
        
        if SPLEETER_AVAILABLE and method in ['spleeter', 'auto']:
            try:
                self.spleeter_separator = SpleeterSeparator('2stems-16kHz')
                self.spleeter_available = True
            except Exception as e:
                print(f"Spleeter initialization failed: {e}")
                self.spleeter_available = False
        else:
            self.spleeter_available = False
        
        # PyTorch model (will be loaded if available)
        self.pytorch_model = PyTorchSeparationModel(n_mels=64)
        self.pytorch_model_trained = False
        
        # Auto-select best available method
        if method == 'auto':
            if self.spleeter_available:
                self.method = 'spleeter'
            else:
                self.method = 'spectral'
        
        print(f"Source separation initialized with method: {self.method}")
        
    def load_pytorch_model(self, model_path: str) -> bool:
        """
        Load trained PyTorch separation model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            True if loaded successfully
        """
        try:
            if os.path.exists(model_path):
                self.pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.pytorch_model.eval()
                self.pytorch_model_trained = True
                print(f"Loaded PyTorch separation model from {model_path}")
                return True
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
        
        return False
    
    def assess_separation_quality(self, original: np.ndarray, 
                                 vocals: np.ndarray, 
                                 instruments: np.ndarray) -> Dict[str, float]:
        """
        Assess quality of separation using various metrics.
        
        Args:
            original: Original audio
            vocals: Separated vocals
            instruments: Separated instruments
            
        Returns:
            Dictionary of quality metrics
        """
        # Energy conservation
        reconstructed = vocals + instruments
        energy_conservation = 1.0 - np.mean((original - reconstructed) ** 2) / (np.mean(original ** 2) + 1e-10)
        
        # Vocal isolation quality (energy in vocal frequency range)
        vocal_freqs = librosa.stft(vocals, n_fft=2048)
        inst_freqs = librosa.stft(instruments, n_fft=2048)
        
        # Vocal frequency range (approx 85-255 Hz fundamental + harmonics)
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        vocal_bins = np.where((freq_bins >= 85) & (freq_bins <= 1000))[0]
        
        vocal_energy_in_vocals = np.mean(np.abs(vocal_freqs[vocal_bins, :]) ** 2)
        vocal_energy_in_instruments = np.mean(np.abs(inst_freqs[vocal_bins, :]) ** 2)
        
        vocal_isolation_ratio = vocal_energy_in_vocals / (vocal_energy_in_vocals + vocal_energy_in_instruments + 1e-10)
        
        # Dynamic range preservation
        original_dynamics = np.std(original)
        vocal_dynamics = np.std(vocals)
        dynamics_preservation = min(vocal_dynamics / (original_dynamics + 1e-10), 1.0)
        
        return {
            'energy_conservation': float(energy_conservation),
            'vocal_isolation_ratio': float(vocal_isolation_ratio),
            'dynamics_preservation': float(dynamics_preservation),
            'overall_quality': float(np.mean([energy_conservation, vocal_isolation_ratio, dynamics_preservation]))
        }
    
    def separate_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Separate audio file into vocals and instruments.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Optional directory to save separated files
            
        Returns:
            Dictionary with separated sources and metadata
        """
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise ValueError(f"Failed to load audio from {audio_path}: {e}")
        
        print(f"Separating {audio_path} using {self.method} method...")
        
        # Perform separation based on method
        if self.method == 'spleeter' and self.spleeter_available:
            sources = self.spleeter_separator.separate(audio, self.sample_rate)
            vocals = sources.get('vocals', np.zeros_like(audio))
            instruments = sources.get('accompaniment', audio - vocals)
            
        elif self.method == 'pytorch' and self.pytorch_model_trained:
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=64, hop_length=252
            )
            mel_db = librosa.power_to_db(mel_spec)
            
            # Normalize
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
            
            # Predict masks
            with torch.no_grad():
                mel_tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)
                vocal_mask, inst_mask = self.pytorch_model(mel_tensor)
                vocal_mask = vocal_mask.squeeze().numpy()
                inst_mask = inst_mask.squeeze().numpy()
            
            # Apply masks to spectrogram
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            
            # Resize masks to match STFT dimensions
            from scipy import ndimage
            vocal_mask_resized = ndimage.zoom(vocal_mask, 
                                            (stft.shape[0] / vocal_mask.shape[0],
                                             stft.shape[1] / vocal_mask.shape[1]))
            inst_mask_resized = ndimage.zoom(inst_mask,
                                           (stft.shape[0] / inst_mask.shape[0],
                                            stft.shape[1] / inst_mask.shape[1]))
            
            # Apply masks
            vocal_stft = stft * vocal_mask_resized
            inst_stft = stft * inst_mask_resized
            
            # Convert back to time domain
            vocals = librosa.istft(vocal_stft, hop_length=512)
            instruments = librosa.istft(inst_stft, hop_length=512)
            
        else:  # Default to spectral masking
            vocals, instruments = self.spectral_separator.separate_vocals_instruments(audio)
        
        # Ensure same length as original
        min_length = min(len(audio), len(vocals), len(instruments))
        audio = audio[:min_length]
        vocals = vocals[:min_length]
        instruments = instruments[:min_length]
        
        # Assess quality
        quality_metrics = self.assess_separation_quality(audio, vocals, instruments)
        
        # Prepare results
        results = {
            'original': audio,
            'vocals': vocals,
            'instruments': instruments,
            'sample_rate': self.sample_rate,
            'method': self.method,
            'quality_metrics': quality_metrics
        }
        
        # Save files if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(audio_path).stem
            
            sf.write(os.path.join(output_dir, f"{base_name}_vocals.wav"), 
                    vocals, self.sample_rate)
            sf.write(os.path.join(output_dir, f"{base_name}_instruments.wav"), 
                    instruments, self.sample_rate)
            
            # Save quality metrics
            with open(os.path.join(output_dir, f"{base_name}_separation_quality.json"), 'w') as f:
                json.dump(quality_metrics, f, indent=2)
            
            print(f"Separated files saved to {output_dir}")
        
        print(f"Separation complete. Quality score: {quality_metrics['overall_quality']:.3f}")
        
        return results
    
    def batch_separate(self, audio_files: List[str], output_dir: str) -> Dict[str, Dict]:
        """
        Separate multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Output directory for separated files
            
        Returns:
            Dictionary with results for each file
        """
        results = {}
        
        for i, audio_file in enumerate(audio_files):
            print(f"\nProcessing {i+1}/{len(audio_files)}: {audio_file}")
            
            try:
                file_output_dir = os.path.join(output_dir, Path(audio_file).stem)
                result = self.separate_audio(audio_file, file_output_dir)
                results[audio_file] = result
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results[audio_file] = {'error': str(e)}
        
        # Save batch summary
        summary = {
            'total_files': len(audio_files),
            'successful': len([r for r in results.values() if 'error' not in r]),
            'failed': len([r for r in results.values() if 'error' in r]),
            'average_quality': np.mean([
                r['quality_metrics']['overall_quality'] 
                for r in results.values() 
                if 'quality_metrics' in r
            ]) if results else 0.0
        }
        
        with open(os.path.join(output_dir, 'batch_separation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBatch separation complete: {summary['successful']}/{summary['total_files']} successful")
        print(f"Average quality score: {summary['average_quality']:.3f}")
        
        return results

def test_source_separation():
    """Test source separation pipeline."""
    print("Testing Source Separation Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    separator = SourceSeparationPipeline(sample_rate=16000, method='auto')
    
    # Test with available audio files
    audio_dir = "data"
    test_files = []
    
    for genre in ['Bangla Folk', 'Jazz', 'Rock']:
        genre_dir = os.path.join(audio_dir, genre)
        if os.path.exists(genre_dir):
            files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3')]
            if files:
                test_files.append(os.path.join(genre_dir, files[0]))  # First file from each genre
    
    if not test_files:
        print("No audio files found for testing.")
        return
    
    # Test separation
    output_dir = "experiments/source_separation_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTesting separation on {len(test_files)} files...")
    results = separator.batch_separate(test_files, output_dir)
    
    # Print results summary
    print("\nSeparation Results:")
    print("-" * 30)
    for file_path, result in results.items():
        filename = Path(file_path).name
        if 'error' not in result:
            quality = result['quality_metrics']['overall_quality']
            print(f"{filename}: Quality = {quality:.3f}")
        else:
            print(f"{filename}: Error - {result['error']}")

if __name__ == "__main__":
    test_source_separation()