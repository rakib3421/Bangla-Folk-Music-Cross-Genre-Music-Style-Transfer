"""
Phase 5.1: Audio Quality Metrics
=================================

This module implements comprehensive objective audio quality metrics for evaluating
the quality of style-transferred music. Includes SNR, PEAQ-inspired metrics,
multi-scale spectral loss, and other perceptual quality measures.

Features:
- Signal-to-Noise Ratio (SNR) calculations
- Perceptual Evaluation of Audio Quality (PEAQ) inspired metrics
- Multi-Scale Spectral Loss analysis
- Spectral distortion measurements
- Perceptual audio quality assessment
- Batch processing for dataset evaluation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import scipy.signal
import scipy.stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("Pyloudnorm not available. Some loudness metrics will be unavailable.")

class AudioQualityMetrics:
    """
    Comprehensive audio quality assessment using objective metrics.
    
    This class implements various audio quality metrics including SNR,
    spectral distortion, perceptual measures, and multi-scale analysis.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio quality metrics calculator.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Initialize loudness meter if available
        if PYLOUDNORM_AVAILABLE:
            self.loudness_meter = pyln.Meter(sample_rate)
        else:
            self.loudness_meter = None
        
        # Perceptual frequency weighting (A-weighting approximation)
        self.freq_weights = self._compute_frequency_weights()
        
    def _compute_frequency_weights(self) -> np.ndarray:
        """Compute A-weighting frequency response for perceptual metrics."""
        # A-weighting filter coefficients (simplified)
        frequencies = np.fft.fftfreq(2048, 1/self.sample_rate)
        frequencies = np.abs(frequencies[:len(frequencies)//2])
        
        # A-weighting formula (simplified)
        # RA(f) = 12194^2 * f^4 / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2) * (f^2 + 737.9^2)) * (f^2 + 12194^2))
        f = frequencies
        f2 = f**2
        
        # Avoid division by zero
        f = np.where(f == 0, 1e-10, f)
        
        numerator = 12194**2 * f**4
        denominator = (
            (f2 + 20.6**2) * 
            np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * 
            (f2 + 12194**2)
        )
        
        a_weighting = numerator / (denominator + 1e-10)
        
        # Convert to dB and normalize
        a_weighting_db = 20 * np.log10(a_weighting + 1e-10)
        a_weighting_db = a_weighting_db - np.max(a_weighting_db)
        
        # Convert back to linear scale
        return 10**(a_weighting_db / 20)
    
    def compute_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio between original and processed audio.
        
        Args:
            original: Original audio signal
            processed: Processed audio signal
            
        Returns:
            SNR in dB
        """
        # Ensure same length
        min_length = min(len(original), len(processed))
        original = original[:min_length]
        processed = processed[:min_length]
        
        # Compute signal and noise power
        signal_power = np.mean(original**2)
        noise_power = np.mean((original - processed)**2)
        
        # Avoid division by zero
        if noise_power < 1e-10:
            return 100.0  # Very high SNR
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)
    
    def compute_thd_n(self, audio: np.ndarray, fundamental_freq: float = 440.0) -> float:
        """
        Compute Total Harmonic Distortion + Noise (THD+N).
        
        Args:
            audio: Audio signal
            fundamental_freq: Fundamental frequency for harmonic analysis
            
        Returns:
            THD+N as a ratio (0-1)
        """
        # Compute FFT
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        # Find fundamental frequency bin
        fundamental_bin = np.argmin(np.abs(freqs - fundamental_freq))
        
        # Find harmonic bins (2f, 3f, 4f, 5f)
        harmonic_bins = []
        for h in range(2, 6):  # 2nd to 5th harmonics
            harmonic_freq = fundamental_freq * h
            if harmonic_freq < self.sample_rate / 2:  # Within Nyquist limit
                harmonic_bin = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_bins.append(harmonic_bin)
        
        # Compute THD+N
        fundamental_power = magnitude[fundamental_bin]**2
        
        # Total harmonic power
        harmonic_power = sum(magnitude[bin]**2 for bin in harmonic_bins)
        
        # Total power (excluding DC)
        total_power = np.sum(magnitude[1:]**2)
        
        # Noise + distortion power
        noise_distortion_power = total_power - fundamental_power
        
        # THD+N ratio
        thd_n = np.sqrt(noise_distortion_power / (fundamental_power + 1e-10))
        
        return float(min(thd_n, 1.0))  # Cap at 1.0
    
    def compute_spectral_distortion(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Compute various spectral distortion metrics.
        
        Args:
            original: Original audio signal
            processed: Processed audio signal
            
        Returns:
            Dictionary of spectral distortion metrics
        """
        # Ensure same length
        min_length = min(len(original), len(processed))
        original = original[:min_length]
        processed = processed[:min_length]
        
        # Compute spectrograms
        orig_stft = librosa.stft(original, n_fft=2048, hop_length=512)
        proc_stft = librosa.stft(processed, n_fft=2048, hop_length=512)
        
        orig_magnitude = np.abs(orig_stft)
        proc_magnitude = np.abs(proc_stft)
        
        # Log-spectral distance
        orig_log = np.log(orig_magnitude + 1e-10)
        proc_log = np.log(proc_magnitude + 1e-10)
        log_spectral_distance = np.mean((orig_log - proc_log)**2)
        
        # Spectral convergence
        numerator = np.sum((orig_magnitude - proc_magnitude)**2)
        denominator = np.sum(orig_magnitude**2)
        spectral_convergence = numerator / (denominator + 1e-10)
        
        # Mel-scale spectral distortion
        orig_mel = librosa.feature.melspectrogram(S=orig_magnitude**2, sr=self.sample_rate)
        proc_mel = librosa.feature.melspectrogram(S=proc_magnitude**2, sr=self.sample_rate)
        
        orig_mel_log = np.log(orig_mel + 1e-10)
        proc_mel_log = np.log(proc_mel + 1e-10)
        mel_spectral_distance = np.mean((orig_mel_log - proc_mel_log)**2)
        
        # Spectral centroid difference
        orig_centroid = librosa.feature.spectral_centroid(S=orig_magnitude, sr=self.sample_rate)
        proc_centroid = librosa.feature.spectral_centroid(S=proc_magnitude, sr=self.sample_rate)
        centroid_difference = np.mean(np.abs(orig_centroid - proc_centroid))
        
        return {
            'log_spectral_distance': float(log_spectral_distance),
            'spectral_convergence': float(spectral_convergence),
            'mel_spectral_distance': float(mel_spectral_distance),
            'spectral_centroid_difference': float(centroid_difference)
        }
    
    def compute_peaq_inspired_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Compute PEAQ-inspired perceptual audio quality metrics.
        
        Args:
            original: Original audio signal
            processed: Processed audio signal
            
        Returns:
            Dictionary of PEAQ-inspired metrics
        """
        # Ensure same length
        min_length = min(len(original), len(processed))
        original = original[:min_length]
        processed = processed[:min_length]
        
        metrics = {}
        
        # Loudness difference
        if self.loudness_meter is not None:
            try:
                orig_loudness = self.loudness_meter.integrated_loudness(original)
                proc_loudness = self.loudness_meter.integrated_loudness(processed)
                metrics['loudness_difference'] = abs(orig_loudness - proc_loudness)
            except:
                metrics['loudness_difference'] = 0.0
        else:
            # Simple RMS-based loudness
            orig_rms = np.sqrt(np.mean(original**2))
            proc_rms = np.sqrt(np.mean(processed**2))
            metrics['loudness_difference'] = abs(20 * np.log10(orig_rms / (proc_rms + 1e-10)))
        
        # Roughness (amplitude modulation detection)
        orig_roughness = self._compute_roughness(original)
        proc_roughness = self._compute_roughness(processed)
        metrics['roughness_difference'] = abs(orig_roughness - proc_roughness)
        
        # Sharpness (high-frequency content)
        orig_sharpness = self._compute_sharpness(original)
        proc_sharpness = self._compute_sharpness(processed)
        metrics['sharpness_difference'] = abs(orig_sharpness - proc_sharpness)
        
        # Fluctuation strength (low-frequency modulation)
        orig_fluctuation = self._compute_fluctuation_strength(original)
        proc_fluctuation = self._compute_fluctuation_strength(processed)
        metrics['fluctuation_difference'] = abs(orig_fluctuation - proc_fluctuation)
        
        # Tonality (harmonic content)
        orig_tonality = self._compute_tonality(original)
        proc_tonality = self._compute_tonality(processed)
        metrics['tonality_difference'] = abs(orig_tonality - proc_tonality)
        
        return metrics
    
    def _compute_roughness(self, audio: np.ndarray) -> float:
        """Compute roughness (amplitude modulation in 15-300 Hz range)."""
        # Extract envelope using Hilbert transform
        analytic_signal = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Compute modulation spectrum
        mod_spectrum = np.abs(np.fft.fft(envelope))
        mod_freqs = np.fft.fftfreq(len(envelope), 1/self.sample_rate)
        
        # Focus on roughness frequency range (15-300 Hz)
        roughness_mask = (np.abs(mod_freqs) >= 15) & (np.abs(mod_freqs) <= 300)
        roughness_energy = np.sum(mod_spectrum[roughness_mask]**2)
        
        # Normalize by total envelope energy
        total_energy = np.sum(envelope**2)
        
        return float(roughness_energy / (total_energy + 1e-10))
    
    def _compute_sharpness(self, audio: np.ndarray) -> float:
        """Compute sharpness (high-frequency content)."""
        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(audio))**2
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Focus on positive frequencies
        pos_mask = freqs >= 0
        spectrum = spectrum[pos_mask]
        freqs = freqs[pos_mask]
        
        # Weight by frequency (higher frequencies contribute more to sharpness)
        freq_weights = freqs / (self.sample_rate / 2)  # Normalize to [0, 1]
        weighted_spectrum = spectrum * freq_weights
        
        # Sharpness as weighted spectral centroid
        sharpness = np.sum(weighted_spectrum) / (np.sum(spectrum) + 1e-10)
        
        return float(sharpness)
    
    def _compute_fluctuation_strength(self, audio: np.ndarray) -> float:
        """Compute fluctuation strength (low-frequency modulation)."""
        # Extract envelope
        analytic_signal = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Compute modulation spectrum
        mod_spectrum = np.abs(np.fft.fft(envelope))
        mod_freqs = np.fft.fftfreq(len(envelope), 1/self.sample_rate)
        
        # Focus on fluctuation frequency range (0.5-20 Hz)
        fluctuation_mask = (np.abs(mod_freqs) >= 0.5) & (np.abs(mod_freqs) <= 20)
        fluctuation_energy = np.sum(mod_spectrum[fluctuation_mask]**2)
        
        # Normalize by total envelope energy
        total_energy = np.sum(envelope**2)
        
        return float(fluctuation_energy / (total_energy + 1e-10))
    
    def _compute_tonality(self, audio: np.ndarray) -> float:
        """Compute tonality (harmonic vs noise content)."""
        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(audio))**2
        
        # Find peaks (potential harmonics)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
        
        # Compute harmonic energy vs total energy
        if len(peaks) > 0:
            harmonic_energy = np.sum(spectrum[peaks])
            total_energy = np.sum(spectrum)
            tonality = harmonic_energy / total_energy
        else:
            tonality = 0.0
        
        return float(tonality)

class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-Scale Spectral Loss for comprehensive spectral analysis.
    
    This loss function computes spectral differences at multiple time-frequency
    resolutions to capture both fine-grained and coarse spectral features.
    """
    
    def __init__(self, sample_rate: int = 16000, scales: List[int] = None):
        """
        Initialize multi-scale spectral loss.
        
        Args:
            sample_rate: Audio sample rate
            scales: List of FFT sizes for different scales
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        
        if scales is None:
            # Default scales: fine to coarse temporal resolution
            self.scales = [512, 1024, 2048, 4096]
        else:
            self.scales = scales
        
        # Mel-scale parameters for each scale
        self.mel_params = []
        for scale in self.scales:
            n_mels = min(80, scale // 16)  # Adaptive mel bins
            self.mel_params.append({
                'n_fft': scale,
                'hop_length': scale // 4,
                'n_mels': n_mels
            })
    
    def compute_spectral_loss(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                             n_fft: int, hop_length: int) -> torch.Tensor:
        """Compute spectral loss for a single scale."""
        # Compute STFTs
        orig_stft = torch.stft(
            original, n_fft=n_fft, hop_length=hop_length, 
            window=torch.hann_window(n_fft, device=original.device),
            return_complex=True
        )
        recon_stft = torch.stft(
            reconstructed, n_fft=n_fft, hop_length=hop_length,
            window=torch.hann_window(n_fft, device=reconstructed.device),
            return_complex=True
        )
        
        # Magnitude spectra
        orig_mag = torch.abs(orig_stft)
        recon_mag = torch.abs(recon_stft)
        
        # L1 and L2 losses in linear scale
        l1_loss = F.l1_loss(recon_mag, orig_mag)
        l2_loss = F.mse_loss(recon_mag, orig_mag)
        
        # Log-magnitude loss
        orig_log = torch.log(orig_mag + 1e-7)
        recon_log = torch.log(recon_mag + 1e-7)
        log_loss = F.l1_loss(recon_log, orig_log)
        
        return l1_loss + l2_loss + log_loss
    
    def compute_mel_loss(self, original: torch.Tensor, reconstructed: torch.Tensor,
                        n_fft: int, hop_length: int, n_mels: int) -> torch.Tensor:
        """Compute mel-scale spectral loss."""
        # Compute mel spectrograms
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(original.device)
        
        orig_mel = mel_transform(original)
        recon_mel = mel_transform(reconstructed)
        
        # Log-mel loss
        orig_log_mel = torch.log(orig_mel + 1e-7)
        recon_log_mel = torch.log(recon_mel + 1e-7)
        
        return F.l1_loss(recon_log_mel, orig_log_mel)
    
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale spectral loss.
        
        Args:
            original: Original audio tensor [B, T] or [B, C, T]
            reconstructed: Reconstructed audio tensor [B, T] or [B, C, T]
            
        Returns:
            Dictionary of loss components
        """
        # Ensure 2D tensors [B, T]
        if original.dim() == 3:
            original = original.squeeze(1)
        if reconstructed.dim() == 3:
            reconstructed = reconstructed.squeeze(1)
        
        # Ensure same length
        min_length = min(original.size(-1), reconstructed.size(-1))
        original = original[..., :min_length]
        reconstructed = reconstructed[..., :min_length]
        
        losses = {}
        total_loss = 0.0
        
        # Compute loss at each scale
        for i, (scale, mel_param) in enumerate(zip(self.scales, self.mel_params)):
            # Skip if audio is too short for this scale
            if min_length < scale:
                continue
            
            # Spectral loss
            spectral_loss = self.compute_spectral_loss(
                original, reconstructed, scale, mel_param['hop_length']
            )
            losses[f'spectral_loss_scale_{i}'] = spectral_loss
            
            # Mel loss
            try:
                mel_loss = self.compute_mel_loss(
                    original, reconstructed, scale, 
                    mel_param['hop_length'], mel_param['n_mels']
                )
                losses[f'mel_loss_scale_{i}'] = mel_loss
            except:
                # Fallback if torchaudio not available
                mel_loss = spectral_loss * 0.5
                losses[f'mel_loss_scale_{i}'] = mel_loss
            
            # Weight by scale (finer scales get higher weight)
            scale_weight = 1.0 / (i + 1)
            total_loss += scale_weight * (spectral_loss + mel_loss)
        
        losses['total_multiscale_loss'] = total_loss
        
        return losses

class AudioQualityEvaluator:
    """
    Comprehensive audio quality evaluator combining all metrics.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio quality evaluator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.quality_metrics = AudioQualityMetrics(sample_rate)
        self.multiscale_loss = MultiScaleSpectralLoss(sample_rate)
        
    def evaluate_pair(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a pair of audio signals (original vs processed).
        
        Args:
            original: Original audio signal
            processed: Processed audio signal
            
        Returns:
            Dictionary of all quality metrics
        """
        results = {}
        
        # Basic quality metrics
        results['snr_db'] = self.quality_metrics.compute_snr(original, processed)
        results['thd_n'] = self.quality_metrics.compute_thd_n(processed)
        
        # Spectral distortion metrics
        spectral_metrics = self.quality_metrics.compute_spectral_distortion(original, processed)
        results.update(spectral_metrics)
        
        # PEAQ-inspired metrics
        peaq_metrics = self.quality_metrics.compute_peaq_inspired_metrics(original, processed)
        results.update(peaq_metrics)
        
        # Multi-scale spectral loss
        try:
            orig_tensor = torch.FloatTensor(original).unsqueeze(0)
            proc_tensor = torch.FloatTensor(processed).unsqueeze(0)
            
            with torch.no_grad():
                multiscale_losses = self.multiscale_loss(orig_tensor, proc_tensor)
                
            for key, value in multiscale_losses.items():
                if isinstance(value, torch.Tensor):
                    results[f'multiscale_{key}'] = float(value.item())
        except Exception as e:
            print(f"Warning: Multi-scale loss computation failed: {e}")
            results['multiscale_total_multiscale_loss'] = 0.0
        
        # Overall quality score (0-1, higher is better)
        results['overall_quality_score'] = self._compute_overall_quality(results)
        
        return results
    
    def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score from individual metrics."""
        # Normalize and weight different metrics
        score = 0.0
        
        # SNR contribution (0-1, higher is better)
        snr = metrics.get('snr_db', 0.0)
        snr_score = min(1.0, max(0.0, (snr + 10) / 40))  # -10 to 30 dB range
        score += 0.3 * snr_score
        
        # THD+N contribution (0-1, lower is better)
        thd_n = metrics.get('thd_n', 1.0)
        thd_score = 1.0 - min(1.0, thd_n)
        score += 0.2 * thd_score
        
        # Spectral distortion contribution
        log_spectral_dist = metrics.get('log_spectral_distance', 10.0)
        spectral_score = max(0.0, 1.0 - log_spectral_dist / 10.0)
        score += 0.2 * spectral_score
        
        # Loudness difference contribution
        loudness_diff = metrics.get('loudness_difference', 10.0)
        loudness_score = max(0.0, 1.0 - loudness_diff / 10.0)
        score += 0.15 * loudness_score
        
        # Multi-scale loss contribution
        multiscale_loss = metrics.get('multiscale_total_multiscale_loss', 10.0)
        multiscale_score = max(0.0, 1.0 - multiscale_loss / 10.0)
        score += 0.15 * multiscale_score
        
        return min(1.0, max(0.0, score))
    
    def evaluate_batch(self, original_list: List[np.ndarray], 
                      processed_list: List[np.ndarray]) -> Dict[str, List[float]]:
        """
        Evaluate a batch of audio pairs.
        
        Args:
            original_list: List of original audio signals
            processed_list: List of processed audio signals
            
        Returns:
            Dictionary with lists of metrics for each pair
        """
        batch_results = {}
        
        for orig, proc in zip(original_list, processed_list):
            pair_results = self.evaluate_pair(orig, proc)
            
            for metric, value in pair_results.items():
                if metric not in batch_results:
                    batch_results[metric] = []
                batch_results[metric].append(value)
        
        return batch_results

def test_audio_quality_metrics():
    """Test audio quality metrics with synthetic data."""
    print("Testing Audio Quality Metrics")
    print("=" * 40)
    
    # Create test signals
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Original signal (clean sine wave)
    freq = 440.0  # A4
    original = np.sin(2 * np.pi * freq * t) * 0.5
    
    # Processed signals with different types of degradation
    test_cases = {
        'clean': original.copy(),
        'noisy': original + np.random.normal(0, 0.05, len(original)),
        'distorted': np.tanh(original * 3) * 0.5,
        'low_pass': scipy.signal.filtfilt(*scipy.signal.butter(4, 2000, fs=sample_rate), original),
        'compressed': np.sign(original) * np.abs(original)**0.5 * 0.5
    }
    
    # Initialize evaluator
    evaluator = AudioQualityEvaluator(sample_rate)
    
    print(f"Evaluating {len(test_cases)} test cases...")
    
    # Evaluate each test case
    results = {}
    for name, processed in test_cases.items():
        print(f"\nEvaluating: {name}")
        
        result = evaluator.evaluate_pair(original, processed)
        results[name] = result
        
        # Print key metrics
        print(f"  SNR: {result['snr_db']:.2f} dB")
        print(f"  THD+N: {result['thd_n']:.4f}")
        print(f"  Overall Quality: {result['overall_quality_score']:.3f}")
        print(f"  Log Spectral Distance: {result['log_spectral_distance']:.4f}")
    
    # Print comparative analysis
    print(f"\n{'='*40}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*40}")
    
    print(f"{'Test Case':<12} {'SNR (dB)':<10} {'THD+N':<8} {'Quality':<8} {'Spectral Dist':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<12} {result['snr_db']:<10.2f} {result['thd_n']:<8.4f} "
              f"{result['overall_quality_score']:<8.3f} {result['log_spectral_distance']:<12.4f}")
    
    print(f"\n✅ Audio Quality Metrics Testing Complete!")
    print(f"   Best Quality: {max(results.keys(), key=lambda k: results[k]['overall_quality_score'])}")
    print(f"   Highest SNR: {max(results.keys(), key=lambda k: results[k]['snr_db'])}")
    
    return results

# Test PyTorch availability for multi-scale loss
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Torchaudio not available. Some multi-scale features will be limited.")

if __name__ == "__main__":
    test_audio_quality_metrics()