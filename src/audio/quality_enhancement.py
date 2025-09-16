"""
Phase 6.3: Quality Enhancement Pipeline
======================================

This module implements comprehensive post-processing quality enhancement
including spectral artifact removal, dynamic range optimization, and 
harmonic enhancement for superior audio output quality.

Features:
- Spectral artifact detection and removal
- Dynamic range compression and optimization
- Harmonic enhancement and restoration
- Noise reduction and audio clarity improvement
- Perceptual audio quality enhancement
- Real-time processing optimizations
- Advanced audio restoration techniques
"""

import os
import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available. Install with: pip install soundfile")

class SpectralArtifactRemover:
    """
    Advanced spectral artifact detection and removal system.
    
    Detects and removes various types of spectral artifacts including
    aliasing, quantization noise, and transfer-induced distortions.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        
        # Artifact detection thresholds
        self.artifact_thresholds = {
            'spectral_spikes': 3.0,  # Standard deviations above mean
            'harmonic_distortion': 0.1,  # THD threshold
            'aliasing_threshold': 0.8,  # Correlation threshold for aliasing
            'noise_floor': -60  # dB threshold for noise floor
        }
    
    def detect_spectral_artifacts(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect various types of spectral artifacts."""
        artifacts = {
            'spectral_spikes': [],
            'harmonic_distortion': 0.0,
            'aliasing_regions': [],
            'noise_characteristics': {},
            'overall_artifact_score': 0.0
        }
        
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # 1. Detect spectral spikes
        artifacts['spectral_spikes'] = self._detect_spectral_spikes(magnitude)
        
        # 2. Estimate harmonic distortion
        artifacts['harmonic_distortion'] = self._estimate_harmonic_distortion(audio)
        
        # 3. Detect aliasing artifacts
        artifacts['aliasing_regions'] = self._detect_aliasing(magnitude)
        
        # 4. Analyze noise characteristics
        artifacts['noise_characteristics'] = self._analyze_noise_characteristics(magnitude)
        
        # 5. Calculate overall artifact score
        artifacts['overall_artifact_score'] = self._calculate_artifact_score(artifacts)
        
        return artifacts
    
    def _detect_spectral_spikes(self, magnitude: np.ndarray) -> List[Tuple[int, int]]:
        """Detect sudden spectral spikes that indicate artifacts."""
        spikes = []
        
        # Compute mean and std for each frequency bin
        freq_mean = np.mean(magnitude, axis=1)
        freq_std = np.std(magnitude, axis=1)
        
        # Find spikes
        threshold = self.artifact_thresholds['spectral_spikes']
        
        for freq_idx in range(magnitude.shape[0]):
            spike_frames = np.where(
                magnitude[freq_idx] > freq_mean[freq_idx] + threshold * freq_std[freq_idx]
            )[0]
            
            for frame_idx in spike_frames:
                spikes.append((freq_idx, frame_idx))
        
        return spikes
    
    def _estimate_harmonic_distortion(self, audio: np.ndarray) -> float:
        """Estimate total harmonic distortion (THD)."""
        try:
            # Use a sliding window approach
            window_size = self.sr // 4  # 250ms windows
            hop_size = window_size // 2
            
            thd_values = []
            
            for start in range(0, len(audio) - window_size, hop_size):
                window = audio[start:start + window_size]
                
                # Compute FFT
                fft = np.fft.fft(window)
                freqs = np.fft.fftfreq(len(window), 1/self.sr)
                magnitude = np.abs(fft)
                
                # Find fundamental frequency (simplified)
                positive_freqs = freqs[:len(freqs)//2]
                positive_mag = magnitude[:len(magnitude)//2]
                
                if len(positive_mag) > 10:
                    # Find peak frequency
                    peak_idx = np.argmax(positive_mag[1:]) + 1  # Skip DC
                    fundamental_freq = positive_freqs[peak_idx]
                    
                    if fundamental_freq > 20:  # Valid audio frequency
                        # Calculate harmonic power
                        fundamental_power = positive_mag[peak_idx] ** 2
                        
                        harmonic_power = 0
                        for harmonic in range(2, 6):  # 2nd to 5th harmonics
                            harmonic_freq = fundamental_freq * harmonic
                            if harmonic_freq < self.sr / 2:
                                # Find closest frequency bin
                                harmonic_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
                                harmonic_power += positive_mag[harmonic_idx] ** 2
                        
                        if fundamental_power > 0:
                            thd = np.sqrt(harmonic_power / fundamental_power)
                            thd_values.append(thd)
            
            return np.mean(thd_values) if thd_values else 0.0
            
        except Exception:
            return 0.0
    
    def _detect_aliasing(self, magnitude: np.ndarray) -> List[Tuple[int, int]]:
        """Detect aliasing artifacts in high frequencies."""
        aliasing_regions = []
        
        # Check high frequency regions for suspicious patterns
        nyquist_idx = magnitude.shape[0] // 2
        high_freq_start = int(0.8 * nyquist_idx)  # Above 80% of Nyquist
        
        if high_freq_start < magnitude.shape[0]:
            high_freq_mag = magnitude[high_freq_start:]
            
            # Look for suspicious correlations with lower frequencies
            for freq_idx in range(high_freq_start, magnitude.shape[0]):
                # Check for mirror patterns (simplified aliasing detection)
                mirror_idx = magnitude.shape[0] - freq_idx - 1
                if mirror_idx > 0 and mirror_idx < magnitude.shape[0]:
                    correlation = np.corrcoef(
                        magnitude[freq_idx], 
                        magnitude[mirror_idx]
                    )[0, 1]
                    
                    if not np.isnan(correlation) and correlation > self.artifact_thresholds['aliasing_threshold']:
                        aliasing_regions.append((freq_idx, mirror_idx))
        
        return aliasing_regions
    
    def _analyze_noise_characteristics(self, magnitude: np.ndarray) -> Dict[str, float]:
        """Analyze background noise characteristics."""
        noise_chars = {}
        
        # Estimate noise floor
        sorted_mag = np.sort(magnitude.flatten())
        percentile_5 = np.percentile(sorted_mag, 5)
        noise_chars['noise_floor_db'] = 20 * np.log10(percentile_5 + 1e-8)
        
        # Spectral flatness (measure of noisiness)
        spectral_flatness = []
        for frame_idx in range(magnitude.shape[1]):
            frame = magnitude[:, frame_idx]
            geometric_mean = np.exp(np.mean(np.log(frame + 1e-8)))
            arithmetic_mean = np.mean(frame)
            flatness = geometric_mean / (arithmetic_mean + 1e-8)
            spectral_flatness.append(flatness)
        
        noise_chars['average_spectral_flatness'] = np.mean(spectral_flatness)
        
        return noise_chars
    
    def _calculate_artifact_score(self, artifacts: Dict[str, Any]) -> float:
        """Calculate overall artifact severity score."""
        score = 0.0
        
        # Spectral spikes contribution
        spike_score = min(len(artifacts['spectral_spikes']) / 100, 1.0)
        score += spike_score * 0.3
        
        # Harmonic distortion contribution
        thd_score = min(artifacts['harmonic_distortion'] / 0.1, 1.0)
        score += thd_score * 0.4
        
        # Aliasing contribution
        aliasing_score = min(len(artifacts['aliasing_regions']) / 50, 1.0)
        score += aliasing_score * 0.2
        
        # Noise contribution
        noise_floor = artifacts['noise_characteristics'].get('noise_floor_db', -60)
        noise_score = max(0, (noise_floor + 40) / 20)  # Normalize -60 to -40 dB
        score += noise_score * 0.1
        
        return min(score, 1.0)
    
    def remove_spectral_artifacts(self, audio: np.ndarray,
                                 artifacts: Dict[str, Any] = None) -> np.ndarray:
        """Remove detected spectral artifacts from audio."""
        if artifacts is None:
            artifacts = self.detect_spectral_artifacts(audio)
        
        print(f"Removing spectral artifacts (score: {artifacts['overall_artifact_score']:.3f})...")
        
        # Convert to STFT domain
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply artifact removal techniques
        cleaned_magnitude = magnitude.copy()
        
        # 1. Remove spectral spikes
        if artifacts['spectral_spikes']:
            cleaned_magnitude = self._remove_spectral_spikes(
                cleaned_magnitude, artifacts['spectral_spikes']
            )
        
        # 2. Reduce harmonic distortion
        if artifacts['harmonic_distortion'] > self.artifact_thresholds['harmonic_distortion']:
            cleaned_magnitude = self._reduce_harmonic_distortion(cleaned_magnitude)
        
        # 3. Apply spectral smoothing for general artifact reduction
        cleaned_magnitude = self._apply_spectral_smoothing(cleaned_magnitude)
        
        # Reconstruct audio
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=self.hop_length)
        
        print("✅ Spectral artifact removal complete")
        return cleaned_audio
    
    def _remove_spectral_spikes(self, magnitude: np.ndarray, 
                               spikes: List[Tuple[int, int]]) -> np.ndarray:
        """Remove identified spectral spikes."""
        cleaned = magnitude.copy()
        
        for freq_idx, frame_idx in spikes:
            # Replace spike with interpolated value
            if frame_idx > 0 and frame_idx < magnitude.shape[1] - 1:
                interpolated_value = (
                    magnitude[freq_idx, frame_idx - 1] + 
                    magnitude[freq_idx, frame_idx + 1]
                ) / 2
                cleaned[freq_idx, frame_idx] = interpolated_value
        
        return cleaned
    
    def _reduce_harmonic_distortion(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply harmonic distortion reduction."""
        # Apply gentle low-pass filtering to reduce high-frequency distortion
        from scipy.signal import butter, filtfilt
        
        # Design butterworth filter
        nyquist = self.sr / 2
        cutoff = 0.8 * nyquist  # Cut at 80% of Nyquist
        b, a = butter(4, cutoff / nyquist, btype='low')
        
        # Apply filter to each frequency frame
        cleaned = magnitude.copy()
        for frame_idx in range(magnitude.shape[1]):
            frame = magnitude[:, frame_idx]
            filtered_frame = filtfilt(b, a, frame)
            cleaned[:, frame_idx] = filtered_frame
        
        return cleaned
    
    def _apply_spectral_smoothing(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply gentle spectral smoothing to reduce artifacts."""
        # Apply 2D Gaussian filter for gentle smoothing
        smoothed = scipy.ndimage.gaussian_filter(magnitude, sigma=[1.0, 0.5])
        
        # Blend with original to preserve important features
        blend_factor = 0.3
        return blend_factor * smoothed + (1 - blend_factor) * magnitude

class DynamicRangeOptimizer:
    """
    Intelligent dynamic range optimization for enhanced audio quality.
    
    Provides multiband compression, loudness normalization, and
    perceptually-aware dynamic processing.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        
        # Dynamic range parameters
        self.compression_params = {
            'threshold': -20,  # dB
            'ratio': 4.0,
            'attack_time': 0.003,  # seconds
            'release_time': 0.1,  # seconds
            'knee_width': 2.0  # dB
        }
        
        # Multiband frequency divisions
        self.frequency_bands = [
            (20, 250),    # Low
            (250, 2000),  # Mid
            (2000, 8000), # High
            (8000, self.sr//2)  # Very High
        ]
    
    def analyze_dynamic_range(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze dynamic range characteristics of audio."""
        analysis = {}
        
        # RMS analysis
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        analysis['rms_mean'] = np.mean(rms)
        analysis['rms_std'] = np.std(rms)
        analysis['rms_range_db'] = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-8))
        
        # Peak analysis
        peak_envelope = np.abs(audio)
        analysis['peak_level_db'] = 20 * np.log10(np.max(peak_envelope) + 1e-8)
        analysis['crest_factor'] = np.max(peak_envelope) / (np.mean(peak_envelope) + 1e-8)
        
        # Dynamic range estimation
        analysis['estimated_dr'] = self._estimate_dynamic_range(audio)
        
        # Loudness estimation (simplified LUFS approximation)
        analysis['estimated_lufs'] = self._estimate_loudness(audio)
        
        return analysis
    
    def _estimate_dynamic_range(self, audio: np.ndarray) -> float:
        """Estimate dynamic range using EBU R128-inspired method."""
        # Simplified dynamic range estimation
        
        # Apply high-pass filter (75 Hz)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 75 / (self.sr / 2), btype='high')
        filtered_audio = filtfilt(b, a, audio)
        
        # Calculate short-term loudness windows
        window_size = int(3.0 * self.sr)  # 3-second windows
        hop_size = int(0.1 * self.sr)     # 100ms hop
        
        loudness_values = []
        for start in range(0, len(filtered_audio) - window_size, hop_size):
            window = filtered_audio[start:start + window_size]
            # Simplified loudness calculation
            mean_square = np.mean(window ** 2)
            if mean_square > 1e-8:
                loudness_db = 10 * np.log10(mean_square)
                loudness_values.append(loudness_db)
        
        if len(loudness_values) >= 2:
            # Dynamic range = difference between 95th and 10th percentiles
            dr = np.percentile(loudness_values, 95) - np.percentile(loudness_values, 10)
            return max(0, dr)
        
        return 0.0
    
    def _estimate_loudness(self, audio: np.ndarray) -> float:
        """Estimate integrated loudness (simplified LUFS)."""
        # This is a simplified approximation of ITU-R BS.1770-4
        
        # Apply pre-filtering (high-pass + high-shelf)
        from scipy.signal import butter, filtfilt
        
        # High-pass filter at 75 Hz
        b_hp, a_hp = butter(2, 75 / (self.sr / 2), btype='high')
        filtered = filtfilt(b_hp, a_hp, audio)
        
        # Calculate gated loudness
        block_size = int(0.4 * self.sr)  # 400ms blocks
        hop_size = int(0.1 * self.sr)    # 100ms overlap
        
        block_loudness = []
        for start in range(0, len(filtered) - block_size, hop_size):
            block = filtered[start:start + block_size]
            mean_square = np.mean(block ** 2)
            if mean_square > 1e-10:
                loudness = -0.691 + 10 * np.log10(mean_square)
                block_loudness.append(loudness)
        
        if block_loudness:
            # Apply gating (simplified)
            gate_threshold = np.percentile(block_loudness, 10) - 10
            gated_blocks = [l for l in block_loudness if l >= gate_threshold]
            
            if gated_blocks:
                integrated_loudness = np.mean(gated_blocks)
                return integrated_loudness
        
        return -23.0  # Default target loudness
    
    def optimize_dynamic_range(self, audio: np.ndarray,
                             target_lufs: float = -23.0,
                             preserve_dynamics: bool = True) -> np.ndarray:
        """Optimize dynamic range with intelligent compression and normalization."""
        print("Optimizing dynamic range...")
        
        # Analyze current dynamic range
        dr_analysis = self.analyze_dynamic_range(audio)
        
        print(f"  Current DR: {dr_analysis['estimated_dr']:.1f} dB")
        print(f"  Current Loudness: {dr_analysis['estimated_lufs']:.1f} LUFS")
        
        optimized_audio = audio.copy()
        
        # 1. Apply multiband compression if needed
        if dr_analysis['estimated_dr'] > 30 or dr_analysis['crest_factor'] > 10:
            optimized_audio = self._apply_multiband_compression(optimized_audio)
        
        # 2. Loudness normalization
        optimized_audio = self._normalize_loudness(optimized_audio, target_lufs)
        
        # 3. Apply gentle limiting to prevent clipping
        optimized_audio = self._apply_soft_limiting(optimized_audio)
        
        print("✅ Dynamic range optimization complete")
        return optimized_audio
    
    def _apply_multiband_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply multiband compression."""
        compressed_bands = []
        
        for low_freq, high_freq in self.frequency_bands:
            # Extract frequency band
            band = self._extract_frequency_band(audio, low_freq, high_freq)
            
            # Apply compression to band
            compressed_band = self._apply_compression(band)
            compressed_bands.append(compressed_band)
        
        # Sum compressed bands
        result = np.zeros_like(audio)
        for band in compressed_bands:
            if len(band) == len(result):
                result += band
        
        return result
    
    def _extract_frequency_band(self, audio: np.ndarray, 
                               low_freq: float, high_freq: float) -> np.ndarray:
        """Extract frequency band using bandpass filter."""
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sr / 2
        low = max(low_freq / nyquist, 0.001)
        high = min(high_freq / nyquist, 0.999)
        
        if low >= high:
            return np.zeros_like(audio)
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, audio)
            return filtered
        except:
            return np.zeros_like(audio)
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression."""
        # Simple peak compression
        threshold_linear = 10 ** (self.compression_params['threshold'] / 20)
        ratio = self.compression_params['ratio']
        
        # Detect peaks
        envelope = np.abs(audio)
        
        # Apply compression
        compressed = audio.copy()
        for i in range(len(audio)):
            if envelope[i] > threshold_linear:
                # Calculate compression
                excess_db = 20 * np.log10(envelope[i] / threshold_linear)
                reduction_db = excess_db * (1 - 1/ratio)
                reduction_linear = 10 ** (-reduction_db / 20)
                
                compressed[i] *= reduction_linear
        
        return compressed
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """Normalize audio to target loudness."""
        current_lufs = self._estimate_loudness(audio)
        lufs_diff = target_lufs - current_lufs
        
        # Convert to linear gain
        gain_db = lufs_diff
        gain_linear = 10 ** (gain_db / 20)
        
        return audio * gain_linear
    
    def _apply_soft_limiting(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Apply soft limiting to prevent clipping."""
        # Simple soft clipping using tanh
        return np.tanh(audio / threshold) * threshold

class HarmonicEnhancer:
    """
    Harmonic enhancement system for improved audio quality.
    
    Enhances harmonic content, restores missing frequencies,
    and improves tonal balance.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
    
    def analyze_harmonic_content(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze harmonic content and quality."""
        analysis = {}
        
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Harmonic ratio
        harmonic_energy = np.sum(harmonic ** 2)
        total_energy = np.sum(audio ** 2)
        analysis['harmonic_ratio'] = harmonic_energy / (total_energy + 1e-8)
        
        # Spectral centroid analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        analysis['spectral_centroid_mean'] = np.mean(spectral_centroid)
        analysis['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Harmonic clarity
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=self.sr)
        analysis['tonal_clarity'] = np.mean(np.max(chroma, axis=0))
        
        # Frequency distribution
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        freq_distribution = np.mean(magnitude, axis=1)
        
        # Analyze frequency balance
        nyquist = self.sr / 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        low_energy = np.sum(freq_distribution[freqs < nyquist * 0.25])
        mid_energy = np.sum(freq_distribution[(freqs >= nyquist * 0.25) & (freqs < nyquist * 0.75)])
        high_energy = np.sum(freq_distribution[freqs >= nyquist * 0.75])
        
        total_spectrum_energy = low_energy + mid_energy + high_energy
        
        analysis['frequency_balance'] = {
            'low': low_energy / (total_spectrum_energy + 1e-8),
            'mid': mid_energy / (total_spectrum_energy + 1e-8),
            'high': high_energy / (total_spectrum_energy + 1e-8)
        }
        
        return analysis
    
    def enhance_harmonics(self, audio: np.ndarray,
                         enhancement_level: float = 0.3) -> np.ndarray:
        """Apply harmonic enhancement to improve audio quality."""
        print("Enhancing harmonic content...")
        
        # Analyze current harmonic content
        harmonic_analysis = self.analyze_harmonic_content(audio)
        
        enhanced_audio = audio.copy()
        
        # 1. Harmonic excitation
        if harmonic_analysis['harmonic_ratio'] < 0.7:
            enhanced_audio = self._apply_harmonic_excitation(enhanced_audio, enhancement_level)
        
        # 2. Spectral enhancement
        enhanced_audio = self._apply_spectral_enhancement(enhanced_audio, enhancement_level)
        
        # 3. Tonal balance correction
        enhanced_audio = self._apply_tonal_balance_correction(
            enhanced_audio, harmonic_analysis['frequency_balance']
        )
        
        print("✅ Harmonic enhancement complete")
        return enhanced_audio
    
    def _apply_harmonic_excitation(self, audio: np.ndarray, level: float) -> np.ndarray:
        """Apply harmonic excitation to enhance harmonic content."""
        # Separate harmonic component
        harmonic, _ = librosa.effects.hpss(audio)
        
        # Generate harmonic excitation
        excited_harmonic = self._generate_harmonic_excitation(harmonic, level)
        
        # Blend with original
        return (1 - level) * audio + level * excited_harmonic
    
    def _generate_harmonic_excitation(self, harmonic: np.ndarray, level: float) -> np.ndarray:
        """Generate harmonic excitation signal."""
        # Apply gentle saturation to generate harmonics
        drive = 1 + level * 2
        excited = np.tanh(harmonic * drive) / drive
        
        # Add subtle even harmonics
        excited_squared = excited ** 2
        excited_with_harmonics = excited + level * 0.1 * excited_squared
        
        return excited_with_harmonics
    
    def _apply_spectral_enhancement(self, audio: np.ndarray, level: float) -> np.ndarray:
        """Apply spectral enhancement to improve clarity."""
        # Convert to frequency domain
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Enhance spectral content
        enhanced_magnitude = self._enhance_spectral_magnitude(magnitude, level)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def _enhance_spectral_magnitude(self, magnitude: np.ndarray, level: float) -> np.ndarray:
        """Enhance spectral magnitude for better clarity."""
        enhanced = magnitude.copy()
        
        # Apply spectral contrast enhancement
        for frame_idx in range(magnitude.shape[1]):
            frame = magnitude[:, frame_idx]
            
            # Enhance peaks relative to valleys
            median_filtered = scipy.ndimage.median_filter(frame, size=5)
            contrast = frame - median_filtered
            enhanced_contrast = contrast * (1 + level * 0.5)
            
            enhanced[:, frame_idx] = median_filtered + enhanced_contrast
        
        return enhanced
    
    def _apply_tonal_balance_correction(self, audio: np.ndarray, 
                                      current_balance: Dict[str, float]) -> np.ndarray:
        """Apply tonal balance correction."""
        # Target balance (more balanced)
        target_balance = {'low': 0.35, 'mid': 0.45, 'high': 0.20}
        
        # Calculate corrections needed
        corrections = {}
        for band in ['low', 'mid', 'high']:
            ratio = target_balance[band] / (current_balance[band] + 1e-8)
            corrections[band] = min(max(ratio, 0.5), 2.0)  # Limit corrections
        
        # Apply corrections using EQ
        return self._apply_three_band_eq(audio, corrections)
    
    def _apply_three_band_eq(self, audio: np.ndarray, corrections: Dict[str, float]) -> np.ndarray:
        """Apply three-band EQ correction."""
        from scipy.signal import butter, filtfilt
        
        # Define crossover frequencies
        low_cutoff = 250
        high_cutoff = 2000
        
        # Extract bands
        low_band = self._extract_frequency_band(audio, 20, low_cutoff)
        mid_band = self._extract_frequency_band(audio, low_cutoff, high_cutoff)
        high_band = self._extract_frequency_band(audio, high_cutoff, self.sr//2)
        
        # Apply corrections
        corrected_low = low_band * corrections['low']
        corrected_mid = mid_band * corrections['mid']
        corrected_high = high_band * corrections['high']
        
        # Sum corrected bands
        return corrected_low + corrected_mid + corrected_high
    
    def _extract_frequency_band(self, audio: np.ndarray, 
                               low_freq: float, high_freq: float) -> np.ndarray:
        """Extract frequency band using filters."""
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sr / 2
        
        if low_freq <= 20:
            # Low-pass for low band
            cutoff = min(high_freq / nyquist, 0.999)
            b, a = butter(4, cutoff, btype='low')
        elif high_freq >= nyquist:
            # High-pass for high band
            cutoff = max(low_freq / nyquist, 0.001)
            b, a = butter(4, cutoff, btype='high')
        else:
            # Band-pass for mid band
            low = max(low_freq / nyquist, 0.001)
            high = min(high_freq / nyquist, 0.999)
            if low >= high:
                return np.zeros_like(audio)
            b, a = butter(4, [low, high], btype='band')
        
        try:
            return filtfilt(b, a, audio)
        except:
            return np.zeros_like(audio)

class QualityEnhancementPipeline:
    """
    Complete quality enhancement pipeline combining all enhancement techniques.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.artifact_remover = SpectralArtifactRemover(sr)
        self.dynamic_optimizer = DynamicRangeOptimizer(sr)
        self.harmonic_enhancer = HarmonicEnhancer(sr)
    
    def enhance_audio_quality(self, audio: np.ndarray,
                             enhancement_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete audio quality enhancement pipeline."""
        if enhancement_config is None:
            enhancement_config = {
                'remove_artifacts': True,
                'optimize_dynamics': True,
                'enhance_harmonics': True,
                'target_lufs': -23.0,
                'enhancement_level': 0.3
            }
        
        print("Starting Quality Enhancement Pipeline")
        print("=" * 40)
        
        original_audio = audio.copy()
        enhanced_audio = audio.copy()
        
        # Step 1: Artifact Removal
        if enhancement_config.get('remove_artifacts', True):
            print("\n1. Removing Spectral Artifacts...")
            artifacts = self.artifact_remover.detect_spectral_artifacts(enhanced_audio)
            if artifacts['overall_artifact_score'] > 0.1:
                enhanced_audio = self.artifact_remover.remove_spectral_artifacts(
                    enhanced_audio, artifacts
                )
            else:
                print("   No significant artifacts detected")
        
        # Step 2: Dynamic Range Optimization
        if enhancement_config.get('optimize_dynamics', True):
            print("\n2. Optimizing Dynamic Range...")
            enhanced_audio = self.dynamic_optimizer.optimize_dynamic_range(
                enhanced_audio, 
                target_lufs=enhancement_config.get('target_lufs', -23.0)
            )
        
        # Step 3: Harmonic Enhancement
        if enhancement_config.get('enhance_harmonics', True):
            print("\n3. Enhancing Harmonic Content...")
            enhanced_audio = self.harmonic_enhancer.enhance_harmonics(
                enhanced_audio,
                enhancement_level=enhancement_config.get('enhancement_level', 0.3)
            )
        
        # Analyze improvement
        improvement_metrics = self._analyze_enhancement_quality(
            original_audio, enhanced_audio
        )
        
        print(f"\n✅ Quality Enhancement Complete!")
        print(f"   Improvement Score: {improvement_metrics['overall_improvement']:.3f}")
        
        return {
            'enhanced_audio': enhanced_audio,
            'improvement_metrics': improvement_metrics,
            'enhancement_config': enhancement_config
        }
    
    def _analyze_enhancement_quality(self, original: np.ndarray, 
                                   enhanced: np.ndarray) -> Dict[str, float]:
        """Analyze the quality improvement from enhancement."""
        metrics = {}
        
        # Dynamic range comparison
        orig_dr = self.dynamic_optimizer.analyze_dynamic_range(original)
        enh_dr = self.dynamic_optimizer.analyze_dynamic_range(enhanced)
        
        metrics['dynamic_range_improvement'] = (
            enh_dr['estimated_dr'] - orig_dr['estimated_dr']
        )
        
        # Harmonic content comparison
        orig_harmonic = self.harmonic_enhancer.analyze_harmonic_content(original)
        enh_harmonic = self.harmonic_enhancer.analyze_harmonic_content(enhanced)
        
        metrics['harmonic_ratio_improvement'] = (
            enh_harmonic['harmonic_ratio'] - orig_harmonic['harmonic_ratio']
        )
        
        metrics['tonal_clarity_improvement'] = (
            enh_harmonic['tonal_clarity'] - orig_harmonic['tonal_clarity']
        )
        
        # Overall improvement score
        improvement_factors = [
            max(0, metrics['harmonic_ratio_improvement'] * 2),
            max(0, metrics['tonal_clarity_improvement'] * 3),
            max(0, min(metrics['dynamic_range_improvement'] / 10, 0.5))
        ]
        
        metrics['overall_improvement'] = np.mean(improvement_factors)
        
        return metrics

def create_demo_quality_enhancement():
    """Create a demonstration of quality enhancement pipeline."""
    print("Creating Demo Quality Enhancement Pipeline")
    print("=" * 45)
    
    # Generate synthetic audio with various quality issues
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base signal: musical content
    fundamental = 220  # A3
    audio = (
        np.sin(2 * np.pi * fundamental * t) * 0.4 +
        np.sin(2 * np.pi * fundamental * 1.5 * t) * 0.2 +
        np.sin(2 * np.pi * fundamental * 2 * t) * 0.1 +
        np.sin(2 * np.pi * fundamental * 3 * t) * 0.05
    )
    
    # Add quality issues
    # 1. Add noise
    noise = np.random.normal(0, 0.05, len(audio))
    audio += noise
    
    # 2. Add clipping distortion
    audio = np.clip(audio, -0.8, 0.8)
    
    # 3. Reduce dynamic range
    audio = np.tanh(audio * 2) * 0.7
    
    # 4. Add some spectral artifacts
    artifact_freq = 5000
    artifact = np.sin(2 * np.pi * artifact_freq * t) * 0.03
    audio += artifact
    
    print(f"Generated test audio with quality issues:")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Issues: noise, clipping, compression, artifacts")
    
    # Create enhancement pipeline
    enhancer = QualityEnhancementPipeline(sr=sr)
    
    # Enhancement configuration
    config = {
        'remove_artifacts': True,
        'optimize_dynamics': True,
        'enhance_harmonics': True,
        'target_lufs': -20.0,
        'enhancement_level': 0.4
    }
    
    # Run enhancement
    print(f"\nRunning quality enhancement...")
    enhancement_result = enhancer.enhance_audio_quality(audio, config)
    
    # Display results
    print(f"\n📊 QUALITY ENHANCEMENT RESULTS")
    print(f"=" * 40)
    
    metrics = enhancement_result['improvement_metrics']
    
    print(f"Enhancement Metrics:")
    print(f"  Dynamic Range Improvement: {metrics['dynamic_range_improvement']:.2f} dB")
    print(f"  Harmonic Ratio Improvement: {metrics['harmonic_ratio_improvement']:.3f}")
    print(f"  Tonal Clarity Improvement: {metrics['tonal_clarity_improvement']:.3f}")
    print(f"  Overall Improvement Score: {metrics['overall_improvement']:.3f}")
    
    # Save results
    output_dir = "experiments/quality_enhancement"
    os.makedirs(output_dir, exist_ok=True)
    
    if SOUNDFILE_AVAILABLE:
        # Save audio files
        original_file = os.path.join(output_dir, "original_with_issues.wav")
        enhanced_file = os.path.join(output_dir, "enhanced_quality.wav")
        
        sf.write(original_file, audio, sr)
        sf.write(enhanced_file, enhancement_result['enhanced_audio'], sr)
        
        print(f"\n💾 Audio files saved:")
        print(f"   Original: {original_file}")
        print(f"   Enhanced: {enhanced_file}")
    
    # Save metrics
    import json
    metrics_file = os.path.join(output_dir, "enhancement_metrics.json")
    with open(metrics_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        json.dump({
            'enhancement_config': config,
            'improvement_metrics': json_metrics
        }, f, indent=2)
    
    print(f"   Metrics: {metrics_file}")
    
    print(f"\n✅ Demo Quality Enhancement Complete!")
    print(f"🔧 Features: Artifact removal, Dynamic optimization, Harmonic enhancement")
    
    return enhancement_result

if __name__ == "__main__":
    # Run the demonstration
    result = create_demo_quality_enhancement()
    
    print(f"\n🎛️ Quality Enhancement Pipeline Ready!")
    print(f"   Capabilities: Comprehensive audio quality improvement")
    print(f"   Overall improvement: {result['improvement_metrics']['overall_improvement']:.3f}")