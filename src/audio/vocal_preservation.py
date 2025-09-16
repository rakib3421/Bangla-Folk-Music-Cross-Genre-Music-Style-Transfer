"""
Phase 4.2: Vocal Preservation and Style Adaptation
=================================================

This module implements vocal preservation techniques for cross-genre style transfer,
including timbre modification, pitch correction, and dynamic range adjustment while
maintaining linguistic content and vocal intelligibility.

Features:
- Vocal timbre modification while preserving phonetic content
- Genre-appropriate pitch correction and scaling
- Dynamic range adjustment for different musical styles
- Formant preservation for linguistic integrity
- Vocal style adaptation with quality assessment
"""

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from scipy import ndimage
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced pitch analysis tools
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Parselmouth not available. Using librosa-based pitch analysis.")

class VocalAnalyzer:
    """
    Analyze vocal characteristics including pitch, formants, and timbre.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_pitch(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract pitch information from vocal audio.
        
        Args:
            audio: Vocal audio signal
            
        Returns:
            Dictionary with pitch information
        """
        if PARSELMOUTH_AVAILABLE:
            return self._extract_pitch_praat(audio)
        else:
            return self._extract_pitch_librosa(audio)
    
    def _extract_pitch_praat(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract pitch using Praat (more accurate for vocals)."""
        try:
            # Create Parselmouth sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            
            # Extract pitch
            pitch = call(sound, "To Pitch", 0.0, 75, 600)  # 75-600 Hz range for vocals
            
            # Get pitch values
            pitch_values = call(pitch, "List values in all frames", "Hertz")
            
            # Get time points
            times = call(pitch, "List all frame times")
            
            # Convert to numpy arrays and handle NaN values
            pitch_values = np.array(pitch_values)
            times = np.array(times)
            
            # Replace NaN/undefined with interpolated values
            valid_mask = ~np.isnan(pitch_values) & (pitch_values > 0)
            if np.sum(valid_mask) > 1:
                interp_func = interp1d(times[valid_mask], pitch_values[valid_mask], 
                                     bounds_error=False, fill_value='extrapolate')
                pitch_values_clean = interp_func(times)
            else:
                pitch_values_clean = np.full_like(pitch_values, 200.0)  # Default to 200 Hz
            
            return {
                'f0': pitch_values_clean,
                'times': times,
                'voiced_mask': valid_mask,
                'mean_f0': np.mean(pitch_values_clean[valid_mask]) if np.sum(valid_mask) > 0 else 200.0,
                'f0_std': np.std(pitch_values_clean[valid_mask]) if np.sum(valid_mask) > 0 else 20.0,
                'method': 'praat'
            }
            
        except Exception as e:
            print(f"Praat pitch extraction failed: {e}, falling back to librosa")
            return self._extract_pitch_librosa(audio)
    
    def _extract_pitch_librosa(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract pitch using librosa."""
        # Extract pitch using YIN algorithm
        f0 = librosa.yin(audio, fmin=75, fmax=600, sr=self.sample_rate)
        
        # Create time array
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(f0)), sr=self.sample_rate, hop_length=hop_length)
        
        # Clean pitch contour
        valid_mask = f0 > 0
        if np.sum(valid_mask) > 1:
            # Interpolate missing values
            interp_func = interp1d(times[valid_mask], f0[valid_mask], 
                                 bounds_error=False, fill_value='extrapolate')
            f0_clean = interp_func(times)
            f0_clean = np.maximum(f0_clean, 75)  # Clamp to minimum
            f0_clean = np.minimum(f0_clean, 600)  # Clamp to maximum
        else:
            f0_clean = np.full_like(f0, 200.0)
        
        return {
            'f0': f0_clean,
            'times': times,
            'voiced_mask': valid_mask,
            'mean_f0': np.mean(f0_clean[valid_mask]) if np.sum(valid_mask) > 0 else 200.0,
            'f0_std': np.std(f0_clean[valid_mask]) if np.sum(valid_mask) > 0 else 20.0,
            'method': 'librosa'
        }
    
    def extract_formants(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract formant frequencies (vocal tract resonances).
        
        Args:
            audio: Vocal audio signal
            
        Returns:
            Dictionary with formant information
        """
        # Compute Linear Prediction Coefficients
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Pre-emphasis filter
        pre_emphasized = scipy.signal.lfilter([1, -0.97], [1], audio)
        
        # Frame the signal
        frames = librosa.util.frame(pre_emphasized, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        
        formants = []
        for frame in frames.T:
            if np.sum(frame**2) > 1e-6:  # Only process frames with sufficient energy
                # Apply window
                windowed = frame * np.hanning(len(frame))
                
                # LPC analysis
                try:
                    lpc_order = min(int(self.sample_rate / 1000) + 2, len(windowed) - 1)
                    lpc_coeffs = librosa.lpc(windowed, order=lpc_order)
                    
                    # Find roots of LPC polynomial
                    roots = np.roots(lpc_coeffs)
                    
                    # Convert to frequencies
                    angles = np.angle(roots)
                    frequencies = angles * self.sample_rate / (2 * np.pi)
                    
                    # Filter for formants (positive frequencies, reasonable range)
                    valid_formants = frequencies[(frequencies > 100) & (frequencies < 4000)]
                    valid_formants = np.sort(valid_formants)[:4]  # First 4 formants
                    
                    # Pad if needed
                    while len(valid_formants) < 4:
                        valid_formants = np.append(valid_formants, 0)
                    
                    formants.append(valid_formants[:4])
                    
                except:
                    # Default formant values if analysis fails
                    formants.append([500, 1500, 2500, 3500])
            else:
                formants.append([500, 1500, 2500, 3500])
        
        formants = np.array(formants)
        times = librosa.frames_to_time(np.arange(formants.shape[0]), 
                                     sr=self.sample_rate, hop_length=hop_length)
        
        return {
            'f1': formants[:, 0],  # First formant
            'f2': formants[:, 1],  # Second formant
            'f3': formants[:, 2],  # Third formant
            'f4': formants[:, 3],  # Fourth formant
            'times': times,
            'mean_f1': np.mean(formants[:, 0]),
            'mean_f2': np.mean(formants[:, 1])
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features that characterize vocal timbre.
        
        Args:
            audio: Vocal audio signal
            
        Returns:
            Dictionary with spectral features
        """
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # MFCCs for timbre
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        
        # Time frames
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(spectral_centroid)), 
                                     sr=self.sample_rate, hop_length=hop_length)
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zcr,
            'mfccs': mfccs,
            'spectral_contrast': spectral_contrast,
            'times': times
        }

class VocalStyleAdapter:
    """
    Adapt vocal style while preserving linguistic content.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.analyzer = VocalAnalyzer(sample_rate)
        
        # Genre-specific vocal characteristics
        self.genre_characteristics = {
            'rock': {
                'pitch_range_multiplier': 1.2,  # Wider pitch range
                'pitch_center_shift': 1.1,      # Slightly higher center
                'dynamic_expansion': 1.3,       # More dynamic
                'brightness_boost': 1.2,        # Brighter timbre
                'roughness_factor': 1.5         # More vocal roughness
            },
            'jazz': {
                'pitch_range_multiplier': 1.1,  # Moderate pitch range
                'pitch_center_shift': 0.95,     # Slightly lower center
                'dynamic_expansion': 1.1,       # Controlled dynamics
                'brightness_boost': 0.9,        # Warmer timbre
                'roughness_factor': 1.1         # Subtle roughness
            },
            'folk': {
                'pitch_range_multiplier': 1.0,  # Natural pitch range
                'pitch_center_shift': 1.0,      # Natural center
                'dynamic_expansion': 1.0,       # Natural dynamics
                'brightness_boost': 1.0,        # Natural timbre
                'roughness_factor': 1.0         # Natural roughness
            }
        }
    
    def adapt_pitch_contour(self, vocal_audio: np.ndarray, target_genre: str) -> np.ndarray:
        """
        Adapt pitch contour for target genre while preserving melodic content.
        
        Args:
            vocal_audio: Input vocal audio
            target_genre: Target genre ('rock', 'jazz', 'folk')
            
        Returns:
            Pitch-adapted vocal audio
        """
        # Extract pitch information
        pitch_info = self.analyzer.extract_pitch(vocal_audio)
        f0 = pitch_info['f0']
        times = pitch_info['times']
        
        # Get genre characteristics
        genre_params = self.genre_characteristics.get(target_genre.lower(), 
                                                    self.genre_characteristics['folk'])
        
        # Adapt pitch contour
        mean_f0 = pitch_info['mean_f0']
        f0_centered = f0 - mean_f0
        
        # Apply genre-specific transformations
        f0_adapted = f0_centered * genre_params['pitch_range_multiplier']
        f0_adapted += mean_f0 * genre_params['pitch_center_shift']
        
        # Smooth the pitch contour to avoid artifacts
        f0_adapted = ndimage.gaussian_filter1d(f0_adapted, sigma=1.0)
        
        # Apply pitch shifting using PSOLA-like approach
        adapted_audio = self._apply_pitch_shift(vocal_audio, pitch_info, f0_adapted)
        
        return adapted_audio
    
    def _apply_pitch_shift(self, audio: np.ndarray, original_pitch: Dict, 
                          target_f0: np.ndarray) -> np.ndarray:
        """
        Apply pitch shifting using phase vocoder approach.
        
        Args:
            audio: Input audio
            original_pitch: Original pitch information
            target_f0: Target F0 contour
            
        Returns:
            Pitch-shifted audio
        """
        # Compute STFT
        hop_length = 256
        stft = librosa.stft(audio, hop_length=hop_length, n_fft=1024)
        
        # Compute pitch shift ratios
        original_f0 = original_pitch['f0']
        
        # Interpolate pitch ratios to match STFT frames
        stft_times = librosa.frames_to_time(np.arange(stft.shape[1]), 
                                          sr=self.sample_rate, hop_length=hop_length)
        
        # Interpolate original and target F0 to STFT timebase
        interp_original = np.interp(stft_times, original_pitch['times'], original_f0)
        interp_target = np.interp(stft_times, original_pitch['times'], target_f0)
        
        # Calculate pitch shift ratios
        pitch_ratios = interp_target / (interp_original + 1e-10)
        pitch_ratios = np.clip(pitch_ratios, 0.5, 2.0)  # Limit extreme shifts
        
        # Apply pitch shifting using phase vocoder
        shifted_stft = np.zeros_like(stft)
        phases = np.angle(stft)
        magnitudes = np.abs(stft)
        
        for t in range(stft.shape[1]):
            shift_ratio = pitch_ratios[t]
            
            # Shift frequency bins
            for f in range(stft.shape[0]):
                shifted_f = int(f * shift_ratio)
                if 0 <= shifted_f < stft.shape[0]:
                    shifted_stft[shifted_f, t] += magnitudes[f, t] * np.exp(1j * phases[f, t])
        
        # Convert back to time domain
        shifted_audio = librosa.istft(shifted_stft, hop_length=hop_length)
        
        return shifted_audio
    
    def adapt_timbre(self, vocal_audio: np.ndarray, target_genre: str) -> np.ndarray:
        """
        Adapt vocal timbre for target genre.
        
        Args:
            vocal_audio: Input vocal audio
            target_genre: Target genre
            
        Returns:
            Timbre-adapted vocal audio
        """
        genre_params = self.genre_characteristics.get(target_genre.lower(), 
                                                    self.genre_characteristics['folk'])
        
        # Extract spectral features
        spectral_features = self.analyzer.extract_spectral_features(vocal_audio)
        
        # Compute STFT for processing
        stft = librosa.stft(vocal_audio, hop_length=256, n_fft=1024)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply brightness adjustment
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=1024)
        
        # Create brightness filter
        brightness_boost = genre_params['brightness_boost']
        if brightness_boost != 1.0:
            # Boost/cut high frequencies for brightness
            freq_weights = np.ones_like(freqs)
            high_freq_mask = freqs > 2000  # Above 2kHz
            freq_weights[high_freq_mask] *= brightness_boost
            
            # Apply frequency weighting
            magnitude = magnitude * freq_weights[:, np.newaxis]
        
        # Apply roughness/breathiness
        roughness_factor = genre_params['roughness_factor']
        if roughness_factor != 1.0:
            # Add controlled noise for roughness
            noise_level = 0.02 * (roughness_factor - 1.0)
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, magnitude.shape)
                magnitude += noise * magnitude
        
        # Reconstruct audio
        adapted_stft = magnitude * np.exp(1j * phase)
        adapted_audio = librosa.istft(adapted_stft, hop_length=256)
        
        return adapted_audio
    
    def adapt_dynamics(self, vocal_audio: np.ndarray, target_genre: str) -> np.ndarray:
        """
        Adapt vocal dynamics for target genre.
        
        Args:
            vocal_audio: Input vocal audio
            target_genre: Target genre
            
        Returns:
            Dynamics-adapted vocal audio
        """
        genre_params = self.genre_characteristics.get(target_genre.lower(), 
                                                    self.genre_characteristics['folk'])
        
        expansion_factor = genre_params['dynamic_expansion']
        
        if expansion_factor == 1.0:
            return vocal_audio
        
        # Compute RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=vocal_audio, hop_length=hop_length)[0]
        
        # Apply dynamic range expansion/compression
        mean_rms = np.mean(rms)
        rms_normalized = rms / (mean_rms + 1e-10)
        
        # Apply expansion/compression
        if expansion_factor > 1.0:
            # Expansion: make loud parts louder, quiet parts quieter
            rms_adapted = np.power(rms_normalized, expansion_factor)
        else:
            # Compression: reduce dynamic range
            rms_adapted = np.power(rms_normalized, expansion_factor)
        
        # Renormalize
        rms_adapted = rms_adapted * mean_rms
        
        # Smooth the gain curve
        rms_adapted = ndimage.gaussian_filter1d(rms_adapted, sigma=2.0)
        
        # Apply gain
        times = librosa.frames_to_time(np.arange(len(rms_adapted)), 
                                     sr=self.sample_rate, hop_length=hop_length)
        audio_times = np.linspace(0, len(vocal_audio) / self.sample_rate, len(vocal_audio))
        
        # Interpolate gain to audio length
        gain_curve = np.interp(audio_times, times, rms_adapted / (rms + 1e-10))
        
        # Apply gain with limiting
        adapted_audio = vocal_audio * gain_curve
        
        # Soft limiting to prevent clipping
        adapted_audio = np.tanh(adapted_audio)
        
        return adapted_audio
    
    def preserve_formants(self, vocal_audio: np.ndarray, 
                         adapted_audio: np.ndarray) -> np.ndarray:
        """
        Preserve formant structure to maintain linguistic content.
        
        Args:
            vocal_audio: Original vocal audio
            adapted_audio: Style-adapted audio
            
        Returns:
            Audio with preserved formants
        """
        # Extract formants from original
        original_formants = self.analyzer.extract_formants(vocal_audio)
        
        # Simple formant preservation using spectral envelope matching
        # This is a simplified approach - more sophisticated methods exist
        
        # Compute spectral envelopes
        hop_length = 256
        original_stft = librosa.stft(vocal_audio, hop_length=hop_length)
        adapted_stft = librosa.stft(adapted_audio, hop_length=hop_length)
        
        # Compute spectral envelopes using cepstral smoothing
        original_mag = np.abs(original_stft)
        adapted_mag = np.abs(adapted_stft)
        
        # Simple spectral envelope extraction
        original_envelope = self._extract_spectral_envelope(original_mag)
        adapted_envelope = self._extract_spectral_envelope(adapted_mag)
        
        # Apply original envelope to adapted magnitude
        envelope_ratio = original_envelope / (adapted_envelope + 1e-10)
        
        # Smooth the ratio to avoid artifacts
        envelope_ratio = ndimage.gaussian_filter(envelope_ratio, sigma=1.0)
        
        # Apply envelope correction
        corrected_mag = adapted_mag * envelope_ratio
        corrected_stft = corrected_mag * np.exp(1j * np.angle(adapted_stft))
        
        # Convert back to time domain
        preserved_audio = librosa.istft(corrected_stft, hop_length=hop_length)
        
        return preserved_audio
    
    def _extract_spectral_envelope(self, magnitude: np.ndarray) -> np.ndarray:
        """Extract spectral envelope using cepstral smoothing."""
        # Simple spectral envelope using moving average
        envelope = np.zeros_like(magnitude)
        
        # Apply smoothing across frequency bins
        for t in range(magnitude.shape[1]):
            spectrum = magnitude[:, t]
            # Log domain for better envelope extraction
            log_spectrum = np.log(spectrum + 1e-10)
            # Smooth in log domain
            smoothed_log = ndimage.gaussian_filter1d(log_spectrum, sigma=3.0)
            # Convert back
            envelope[:, t] = np.exp(smoothed_log)
        
        return envelope
    
    def adapt_vocal_style(self, vocal_audio: np.ndarray, 
                         target_genre: str, 
                         preserve_formants: bool = True) -> Dict[str, np.ndarray]:
        """
        Complete vocal style adaptation pipeline.
        
        Args:
            vocal_audio: Input vocal audio
            target_genre: Target genre for adaptation
            preserve_formants: Whether to preserve formant structure
            
        Returns:
            Dictionary with adapted audio and metadata
        """
        print(f"Adapting vocal style for {target_genre} genre...")
        
        # Step 1: Pitch adaptation
        pitch_adapted = self.adapt_pitch_contour(vocal_audio, target_genre)
        
        # Step 2: Timbre adaptation
        timbre_adapted = self.adapt_timbre(pitch_adapted, target_genre)
        
        # Step 3: Dynamic range adaptation
        dynamics_adapted = self.adapt_dynamics(timbre_adapted, target_genre)
        
        # Step 4: Formant preservation (optional)
        if preserve_formants:
            final_adapted = self.preserve_formants(vocal_audio, dynamics_adapted)
        else:
            final_adapted = dynamics_adapted
        
        # Extract features for comparison
        original_features = self.analyzer.extract_spectral_features(vocal_audio)
        adapted_features = self.analyzer.extract_spectral_features(final_adapted)
        
        return {
            'adapted_vocals': final_adapted,
            'intermediate_steps': {
                'pitch_adapted': pitch_adapted,
                'timbre_adapted': timbre_adapted,
                'dynamics_adapted': dynamics_adapted
            },
            'original_features': original_features,
            'adapted_features': adapted_features,
            'adaptation_parameters': self.genre_characteristics.get(
                target_genre.lower(), self.genre_characteristics['folk']
            )
        }

def test_vocal_preservation():
    """Test vocal preservation system."""
    print("Testing Vocal Preservation System")
    print("=" * 50)
    
    # Create test vocal audio (synthetic)
    sample_rate = 16000
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create test vocal signal with pitch modulation
    f0_base = 200  # Base frequency (200 Hz)
    f0_modulation = 20 * np.sin(2 * np.pi * 2 * t)  # Pitch modulation
    f0 = f0_base + f0_modulation
    
    # Generate vocal-like signal with harmonics
    vocal_signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
        vocal_signal += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
    
    # Add some noise for realism
    vocal_signal += 0.05 * np.random.normal(size=len(vocal_signal))
    
    # Normalize
    vocal_signal = vocal_signal / np.max(np.abs(vocal_signal))
    
    # Test vocal style adaptation
    adapter = VocalStyleAdapter(sample_rate)
    
    # Test adaptation to different genres
    test_genres = ['rock', 'jazz', 'folk']
    output_dir = "experiments/vocal_preservation_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    sf.write(os.path.join(output_dir, "original_vocal.wav"), vocal_signal, sample_rate)
    
    print(f"\nTesting vocal adaptation for {len(test_genres)} genres...")
    
    for genre in test_genres:
        print(f"\nAdapting to {genre} style...")
        result = adapter.adapt_vocal_style(vocal_signal, genre)
        
        # Save adapted vocal
        adapted_audio = result['adapted_vocals']
        sf.write(os.path.join(output_dir, f"adapted_vocal_{genre}.wav"), 
                adapted_audio, sample_rate)
        
        # Save adaptation metadata
        metadata = {
            'genre': genre,
            'adaptation_parameters': result['adaptation_parameters'],
            'quality_metrics': {
                'original_spectral_centroid': float(np.mean(result['original_features']['spectral_centroid'])),
                'adapted_spectral_centroid': float(np.mean(result['adapted_features']['spectral_centroid'])),
                'spectral_centroid_change': float(
                    np.mean(result['adapted_features']['spectral_centroid']) / 
                    np.mean(result['original_features']['spectral_centroid'])
                )
            }
        }
        
        with open(os.path.join(output_dir, f"adaptation_metadata_{genre}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Spectral centroid change: {metadata['quality_metrics']['spectral_centroid_change']:.3f}x")
    
    print(f"\nVocal preservation test complete. Results saved to {output_dir}")

if __name__ == "__main__":
    test_vocal_preservation()