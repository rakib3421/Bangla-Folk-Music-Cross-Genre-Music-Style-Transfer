"""
Phase 4.3: Rhythmic Analysis and Consistency Framework
=====================================================

This module implements comprehensive rhythmic analysis including tempo mapping,
beat tracking, and rhythm preservation for cross-genre style transfer while
maintaining traditional folk rhythmic patterns.

Features:
- Multi-algorithm tempo estimation and beat tracking
- Rhythmic pattern extraction and analysis
- Genre-appropriate rhythmic embellishment mapping
- Traditional folk rhythm preservation
- Rhythmic coherence assessment and post-processing
"""

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from scipy import ndimage
from scipy.stats import mode
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import madmom for advanced beat tracking
try:
    import madmom
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    print("Madmom not available. Using librosa-based beat tracking only.")

class TempoAnalyzer:
    """
    Comprehensive tempo analysis using multiple algorithms for robust estimation.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def estimate_tempo_librosa(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Estimate tempo using librosa's tempo estimation.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with tempo information
        """
        # Extract onset strength
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=512
        )
        
        # Estimate tempo
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_envelope, sr=self.sample_rate, hop_length=512,
            start_bpm=60, std_bpm=1.0, tightness=100
        )
        
        # Get beat times
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=512)
        
        # Calculate tempo stability
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            tempo_stability = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
        else:
            tempo_stability = 0.0
        
        return {
            'tempo': float(tempo),
            'beat_times': beat_times,
            'onset_envelope': onset_envelope,
            'tempo_stability': float(tempo_stability),
            'method': 'librosa'
        }
    
    def estimate_tempo_autocorrelation(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Estimate tempo using autocorrelation of onset strength.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with tempo information
        """
        # Extract onset strength
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=512
        )
        
        # Compute autocorrelation
        autocorr = np.correlate(onset_envelope, onset_envelope, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        hop_length = 512
        time_axis = librosa.frames_to_time(np.arange(len(autocorr)), 
                                         sr=self.sample_rate, hop_length=hop_length)
        
        # Look for peaks in reasonable tempo range (60-180 BPM)
        min_period = 60 / 180  # seconds for 180 BPM
        max_period = 60 / 60   # seconds for 60 BPM
        
        valid_indices = np.where((time_axis >= min_period) & (time_axis <= max_period))[0]
        
        if len(valid_indices) > 0:
            valid_autocorr = autocorr[valid_indices]
            valid_times = time_axis[valid_indices]
            
            # Find peak
            peak_idx = np.argmax(valid_autocorr)
            beat_period = valid_times[peak_idx]
            tempo = 60.0 / beat_period
        else:
            tempo = 120.0  # Default tempo
            beat_period = 0.5
        
        return {
            'tempo': float(tempo),
            'beat_period': float(beat_period),
            'autocorrelation': autocorr,
            'time_axis': time_axis,
            'method': 'autocorrelation'
        }
    
    def estimate_tempo_madmom(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Estimate tempo using Madmom (if available).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with tempo information
        """
        if not MADMOM_AVAILABLE:
            return self.estimate_tempo_librosa(audio)
        
        try:
            # Madmom tempo estimation
            proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
            act = madmom.features.beats.RNNBeatProcessor()(audio)
            tempo = proc(act)
            
            # Beat tracking
            beat_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beats = beat_proc(act)
            
            return {
                'tempo': float(tempo[0][0]) if len(tempo) > 0 else 120.0,
                'tempo_strength': float(tempo[0][1]) if len(tempo) > 0 else 1.0,
                'beat_times': beats,
                'method': 'madmom'
            }
            
        except Exception as e:
            print(f"Madmom tempo estimation failed: {e}")
            return self.estimate_tempo_librosa(audio)
    
    def estimate_tempo_ensemble(self, audio: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Ensemble tempo estimation using multiple methods.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with ensemble tempo information
        """
        methods = []
        
        # Librosa method
        librosa_result = self.estimate_tempo_librosa(audio)
        methods.append(('librosa', librosa_result['tempo']))
        
        # Autocorrelation method
        autocorr_result = self.estimate_tempo_autocorrelation(audio)
        methods.append(('autocorr', autocorr_result['tempo']))
        
        # Madmom method (if available)
        if MADMOM_AVAILABLE:
            madmom_result = self.estimate_tempo_madmom(audio)
            methods.append(('madmom', madmom_result['tempo']))
        
        # Ensemble decision
        tempos = [tempo for _, tempo in methods]
        
        # Remove outliers (more than 1.5x or less than 0.67x the median)
        median_tempo = np.median(tempos)
        valid_tempos = [t for t in tempos if 0.67 * median_tempo <= t <= 1.5 * median_tempo]
        
        if valid_tempos:
            final_tempo = np.mean(valid_tempos)
            confidence = len(valid_tempos) / len(tempos)
        else:
            final_tempo = median_tempo
            confidence = 0.5
        
        return {
            'tempo': float(final_tempo),
            'confidence': float(confidence),
            'method_results': dict(methods),
            'beat_times': librosa_result['beat_times'],  # Use librosa beats as default
            'method': 'ensemble'
        }

class BeatTracker:
    """
    Advanced beat tracking with multiple algorithms and genre-specific adaptations.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.tempo_analyzer = TempoAnalyzer(sample_rate)
        
    def track_beats_librosa(self, audio: np.ndarray, tempo: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Track beats using librosa's beat tracker.
        
        Args:
            audio: Input audio signal
            tempo: Optional tempo hint
            
        Returns:
            Dictionary with beat tracking information
        """
        # Extract onset strength
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=512
        )
        
        # Track beats
        if tempo is not None:
            tempo_est, beats = librosa.beat.beat_track(
                onset_envelope=onset_envelope, sr=self.sample_rate, 
                hop_length=512, start_bpm=tempo, tightness=100
            )
        else:
            tempo_est, beats = librosa.beat.beat_track(
                onset_envelope=onset_envelope, sr=self.sample_rate, hop_length=512
            )
        
        # Convert to time
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=512)
        
        # Extract downbeats (simple heuristic)
        if len(beat_times) > 0:
            # Assume 4/4 time signature - every 4th beat is a downbeat
            downbeat_indices = np.arange(0, len(beat_times), 4)
            downbeat_times = beat_times[downbeat_indices]
        else:
            downbeat_times = np.array([])
        
        return {
            'beat_times': beat_times,
            'downbeat_times': downbeat_times,
            'tempo': float(tempo_est),
            'onset_envelope': onset_envelope,
            'method': 'librosa'
        }
    
    def track_beats_madmom(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Track beats using Madmom (if available).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with beat tracking information
        """
        if not MADMOM_AVAILABLE:
            return self.track_beats_librosa(audio)
        
        try:
            # Beat tracking
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            act = madmom.features.beats.RNNBeatProcessor()(audio)
            beats = proc(act)
            
            # Downbeat tracking
            try:
                downbeat_proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
                    beats_per_bar=[4], fps=100  # Assume 4/4 time
                )
                downbeats = downbeat_proc(act)
                downbeat_times = downbeats[:, 0]  # Extract times only
            except:
                # Fallback: estimate downbeats from beats
                if len(beats) > 0:
                    downbeat_indices = np.arange(0, len(beats), 4)
                    downbeat_times = beats[downbeat_indices]
                else:
                    downbeat_times = np.array([])
            
            return {
                'beat_times': beats,
                'downbeat_times': downbeat_times,
                'method': 'madmom'
            }
            
        except Exception as e:
            print(f"Madmom beat tracking failed: {e}")
            return self.track_beats_librosa(audio)
    
    def extract_rhythmic_pattern(self, audio: np.ndarray, beat_times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract rhythmic patterns from audio given beat positions.
        
        Args:
            audio: Input audio signal
            beat_times: Beat time positions
            
        Returns:
            Dictionary with rhythmic pattern information
        """
        if len(beat_times) < 2:
            return {'pattern': np.array([]), 'subdivisions': np.array([])}
        
        # Calculate inter-beat intervals
        beat_intervals = np.diff(beat_times)
        avg_beat_interval = np.mean(beat_intervals)
        
        # Extract onset strength
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=512
        )
        onset_times = librosa.frames_to_time(
            np.arange(len(onset_envelope)), sr=self.sample_rate, hop_length=512
        )
        
        # Find onsets between beats
        rhythmic_patterns = []
        
        for i in range(len(beat_times) - 1):
            beat_start = beat_times[i]
            beat_end = beat_times[i + 1]
            
            # Find onsets in this beat interval
            onset_mask = (onset_times >= beat_start) & (onset_times < beat_end)
            beat_onsets = onset_times[onset_mask]
            beat_onset_strengths = onset_envelope[onset_mask]
            
            # Normalize onset times to beat interval [0, 1]
            if len(beat_onsets) > 0:
                normalized_onsets = (beat_onsets - beat_start) / (beat_end - beat_start)
                
                # Quantize to common subdivisions (16th notes)
                subdivisions = np.round(normalized_onsets * 16) / 16
                
                # Create pattern representation
                pattern = np.zeros(16)  # 16th note resolution
                for onset_time, strength in zip(subdivisions, beat_onset_strengths):
                    idx = int(onset_time * 16)
                    if 0 <= idx < 16:
                        pattern[idx] = max(pattern[idx], strength)
                
                rhythmic_patterns.append(pattern)
        
        if rhythmic_patterns:
            # Average pattern across all beats
            avg_pattern = np.mean(rhythmic_patterns, axis=0)
            pattern_std = np.std(rhythmic_patterns, axis=0)
        else:
            avg_pattern = np.zeros(16)
            pattern_std = np.zeros(16)
        
        return {
            'pattern': avg_pattern,
            'pattern_std': pattern_std,
            'all_patterns': np.array(rhythmic_patterns) if rhythmic_patterns else np.array([]),
            'avg_beat_interval': avg_beat_interval
        }

class RhythmicConstraintSystem:
    """
    System for applying rhythmic constraints during style transfer.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.beat_tracker = BeatTracker(sample_rate)
        
        # Genre-specific rhythmic characteristics
        self.genre_rhythmic_patterns = {
            'folk': {
                'typical_patterns': [
                    np.array([1, 0, 0.5, 0, 1, 0, 0.5, 0, 1, 0, 0.5, 0, 1, 0, 0.5, 0]),  # Simple 4/4
                    np.array([1, 0, 0, 0.3, 0.7, 0, 0, 0.3, 1, 0, 0, 0.3, 0.7, 0, 0, 0.3])  # Folk shuffle
                ],
                'tempo_range': (80, 140),
                'beat_emphasis': 'strong_downbeats',
                'subdivision_preference': 'quarter_eighth'
            },
            'rock': {
                'typical_patterns': [
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),  # Four-on-floor
                    np.array([1, 0, 0.3, 0, 1, 0, 0.3, 0, 1, 0, 0.3, 0, 1, 0, 0.3, 0])  # Rock beat
                ],
                'tempo_range': (100, 180),
                'beat_emphasis': 'backbeat',
                'subdivision_preference': 'eighth_sixteenth'
            },
            'jazz': {
                'typical_patterns': [
                    np.array([1, 0, 0.6, 0, 0.8, 0, 0.6, 0, 1, 0, 0.6, 0, 0.8, 0, 0.6, 0]),  # Swing
                    np.array([1, 0, 0, 0.4, 0.7, 0, 0, 0.4, 1, 0, 0, 0.4, 0.7, 0, 0, 0.4])  # Jazz shuffle
                ],
                'tempo_range': (60, 200),
                'beat_emphasis': 'syncopated',
                'subdivision_preference': 'triplet_swing'
            }
        }
    
    def analyze_rhythmic_content(self, audio: np.ndarray) -> Dict:
        """
        Comprehensive rhythmic analysis of audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with complete rhythmic analysis
        """
        # Tempo estimation
        tempo_info = self.beat_tracker.tempo_analyzer.estimate_tempo_ensemble(audio)
        
        # Beat tracking
        beat_info = self.beat_tracker.track_beats_librosa(audio, tempo_info['tempo'])
        
        # Rhythmic pattern extraction
        pattern_info = self.beat_tracker.extract_rhythmic_pattern(
            audio, beat_info['beat_times']
        )
        
        # Genre classification based on rhythmic features
        genre_scores = self._classify_rhythmic_genre(tempo_info, pattern_info)
        
        return {
            'tempo': tempo_info,
            'beats': beat_info,
            'patterns': pattern_info,
            'genre_scores': genre_scores,
            'rhythmic_stability': self._assess_rhythmic_stability(beat_info),
            'complexity_score': self._calculate_rhythmic_complexity(pattern_info)
        }
    
    def _classify_rhythmic_genre(self, tempo_info: Dict, pattern_info: Dict) -> Dict[str, float]:
        """Classify genre based on rhythmic characteristics."""
        scores = {}
        tempo = tempo_info['tempo']
        pattern = pattern_info['pattern']
        
        for genre, characteristics in self.genre_rhythmic_patterns.items():
            score = 0.0
            
            # Tempo score
            tempo_range = characteristics['tempo_range']
            if tempo_range[0] <= tempo <= tempo_range[1]:
                tempo_score = 1.0
            else:
                # Penalize based on distance from range
                if tempo < tempo_range[0]:
                    tempo_score = max(0, 1 - (tempo_range[0] - tempo) / 50)
                else:
                    tempo_score = max(0, 1 - (tempo - tempo_range[1]) / 50)
            
            score += 0.3 * tempo_score
            
            # Pattern similarity score
            if len(pattern) > 0:
                pattern_scores = []
                for typical_pattern in characteristics['typical_patterns']:
                    # Normalize patterns
                    norm_pattern = pattern / (np.max(pattern) + 1e-10)
                    norm_typical = typical_pattern / (np.max(typical_pattern) + 1e-10)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(norm_pattern, norm_typical)[0, 1]
                    if not np.isnan(correlation):
                        pattern_scores.append(max(0, correlation))
                
                if pattern_scores:
                    score += 0.7 * max(pattern_scores)
            
            scores[genre] = score
        
        return scores
    
    def _assess_rhythmic_stability(self, beat_info: Dict) -> float:
        """Assess rhythmic stability based on beat consistency."""
        beat_times = beat_info['beat_times']
        
        if len(beat_times) < 3:
            return 0.0
        
        # Calculate inter-beat intervals
        intervals = np.diff(beat_times)
        
        # Stability is inverse of coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval > 0:
            stability = 1.0 - (std_interval / mean_interval)
            return max(0.0, min(1.0, stability))
        else:
            return 0.0
    
    def _calculate_rhythmic_complexity(self, pattern_info: Dict) -> float:
        """Calculate rhythmic complexity score."""
        pattern = pattern_info['pattern']
        
        if len(pattern) == 0:
            return 0.0
        
        # Complexity based on number of onset points and syncopation
        num_onsets = np.sum(pattern > 0.1)  # Threshold for significant onsets
        
        # Syncopation score (onsets on weak beats)
        weak_beats = [1, 3, 5, 7, 9, 11, 13, 15]  # Off-beat positions
        syncopation = np.sum(pattern[weak_beats])
        total_energy = np.sum(pattern)
        
        syncopation_ratio = syncopation / (total_energy + 1e-10)
        
        # Combine metrics
        complexity = (num_onsets / 16) * 0.6 + syncopation_ratio * 0.4
        
        return min(1.0, complexity)
    
    def apply_rhythmic_constraints(self, audio: np.ndarray, 
                                 target_genre: str,
                                 preserve_original: float = 0.7) -> np.ndarray:
        """
        Apply rhythmic constraints to audio for target genre.
        
        Args:
            audio: Input audio signal
            target_genre: Target genre for adaptation
            preserve_original: How much of original rhythm to preserve (0-1)
            
        Returns:
            Rhythmically constrained audio
        """
        # Analyze original rhythm
        original_analysis = self.analyze_rhythmic_content(audio)
        
        # Get target genre characteristics
        target_characteristics = self.genre_rhythmic_patterns.get(
            target_genre.lower(), self.genre_rhythmic_patterns['folk']
        )
        
        # Apply tempo adjustment if needed
        current_tempo = original_analysis['tempo']['tempo']
        target_tempo_range = target_characteristics['tempo_range']
        
        if current_tempo < target_tempo_range[0]:
            target_tempo = target_tempo_range[0]
        elif current_tempo > target_tempo_range[1]:
            target_tempo = target_tempo_range[1]
        else:
            target_tempo = current_tempo
        
        # Apply tempo change
        tempo_ratio = target_tempo / current_tempo
        if abs(tempo_ratio - 1.0) > 0.05:  # Only if significant change needed
            audio_resampled = librosa.effects.time_stretch(audio, rate=1/tempo_ratio)
        else:
            audio_resampled = audio
        
        # Apply rhythmic pattern adaptation (simplified approach)
        if preserve_original < 1.0:
            # This is a placeholder for more sophisticated rhythmic adaptation
            # In practice, this would involve complex spectral and temporal manipulation
            adaptation_strength = 1.0 - preserve_original
            
            # Simple rhythmic emphasis adjustment
            beat_info = self.beat_tracker.track_beats_librosa(audio_resampled)
            beat_times = beat_info['beat_times']
            
            if len(beat_times) > 0:
                # Create beat emphasis based on target genre
                audio_emphasized = self._apply_beat_emphasis(
                    audio_resampled, beat_times, target_characteristics, adaptation_strength
                )
            else:
                audio_emphasized = audio_resampled
        else:
            audio_emphasized = audio_resampled
        
        return audio_emphasized
    
    def _apply_beat_emphasis(self, audio: np.ndarray, beat_times: np.ndarray,
                           characteristics: Dict, strength: float) -> np.ndarray:
        """Apply genre-specific beat emphasis."""
        if len(beat_times) == 0:
            return audio
        
        # Create emphasis pattern based on genre
        emphasis_type = characteristics['beat_emphasis']
        
        # Simple implementation: apply gain modulation at beat positions
        emphasis_audio = audio.copy()
        
        for i, beat_time in enumerate(beat_times):
            beat_sample = int(beat_time * self.sample_rate)
            
            # Determine emphasis based on beat type and genre
            if emphasis_type == 'strong_downbeats' and i % 4 == 0:
                # Emphasize downbeats
                emphasis = 1.0 + 0.2 * strength
            elif emphasis_type == 'backbeat' and i % 4 in [1, 3]:
                # Emphasize backbeats (2 and 4)
                emphasis = 1.0 + 0.15 * strength
            elif emphasis_type == 'syncopated':
                # More complex syncopated emphasis
                emphasis = 1.0 + 0.1 * strength * np.random.uniform(0.5, 1.5)
            else:
                emphasis = 1.0
            
            # Apply emphasis in a small window around the beat
            window_size = int(0.05 * self.sample_rate)  # 50ms window
            start_idx = max(0, beat_sample - window_size // 2)
            end_idx = min(len(emphasis_audio), beat_sample + window_size // 2)
            
            # Apply smooth gain modulation
            window = np.hanning(end_idx - start_idx)
            gain_curve = 1.0 + (emphasis - 1.0) * window
            
            emphasis_audio[start_idx:end_idx] *= gain_curve
        
        return emphasis_audio

def test_rhythmic_analysis():
    """Test rhythmic analysis system."""
    print("Testing Rhythmic Analysis Framework")
    print("=" * 50)
    
    # Test with available audio files
    constraint_system = RhythmicConstraintSystem(sample_rate=16000)
    
    # Find test files
    audio_dir = "data"
    test_files = []
    
    for genre in ['Bangla Folk', 'Jazz', 'Rock']:
        genre_dir = os.path.join(audio_dir, genre)
        if os.path.exists(genre_dir):
            files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3')]
            if files:
                test_files.append((genre, os.path.join(genre_dir, files[0])))
    
    if not test_files:
        print("No audio files found for testing.")
        return
    
    # Test rhythmic analysis
    output_dir = "experiments/rhythmic_analysis_test"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for genre, file_path in test_files:
        print(f"\nAnalyzing {genre}: {Path(file_path).name}")
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000, duration=30.0)  # First 30 seconds
            
            # Analyze rhythm
            analysis = constraint_system.analyze_rhythmic_content(audio)
            
            # Save results
            results[genre] = {
                'tempo': analysis['tempo']['tempo'],
                'confidence': analysis['tempo']['confidence'],
                'rhythmic_stability': analysis['rhythmic_stability'],
                'complexity_score': analysis['complexity_score'],
                'genre_scores': analysis['genre_scores']
            }
            
            print(f"  Tempo: {analysis['tempo']['tempo']:.1f} BPM")
            print(f"  Stability: {analysis['rhythmic_stability']:.3f}")
            print(f"  Complexity: {analysis['complexity_score']:.3f}")
            print(f"  Genre scores: {analysis['genre_scores']}")
            
            # Test rhythmic adaptation
            for target_genre in ['rock', 'jazz']:
                if target_genre != genre.lower():
                    adapted_audio = constraint_system.apply_rhythmic_constraints(
                        audio, target_genre, preserve_original=0.8
                    )
                    
                    # Save adapted audio
                    output_filename = f"{genre.lower().replace(' ', '_')}_to_{target_genre}.wav"
                    sf.write(os.path.join(output_dir, output_filename), adapted_audio, 16000)
                    
                    print(f"  Adapted to {target_genre}: saved {output_filename}")
        
        except Exception as e:
            print(f"  Error analyzing {file_path}: {e}")
    
    # Save analysis summary
    with open(os.path.join(output_dir, 'rhythmic_analysis_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRhythmic analysis test complete. Results saved to {output_dir}")

if __name__ == "__main__":
    test_rhythmic_analysis()