"""
Phase 5.4: Musical Evaluation System
===================================

This module implements comprehensive musical evaluation for style-transferred audio,
focusing on rhythm preservation, lyrical intelligibility, musical coherence, and
genre-specific musical characteristics.

Features:
- Rhythm preservation analysis with beat tracking and tempo consistency
- Lyrical intelligibility assessment using speech recognition and clarity metrics
- Musical coherence evaluation through harmonic progression and structure analysis
- Genre-specific musical characteristic evaluation
- Advanced music information retrieval (MIR) techniques
- Multi-dimensional musical quality scoring
"""

import os
import numpy as np
import librosa
import scipy
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import speech_recognition as sr
except ImportError:
    sr = None
    print("Warning: speech_recognition not available. Install with: pip install SpeechRecognition")

try:
    import mir_eval
except ImportError:
    mir_eval = None
    print("Warning: mir_eval not available. Install with: pip install mir_eval")

class RhythmPreservationAnalyzer:
    """
    Analyzes rhythm preservation in style-transferred music.
    
    Evaluates tempo consistency, beat alignment, rhythmic pattern preservation,
    and groove characteristics between original and transferred audio.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        self.frame_time = self.hop_length / self.sr
    
    def extract_rhythmic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive rhythmic features from audio."""
        features = {}
        
        # 1. Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        beat_times = librosa.times_like(beats, sr=self.sr, hop_length=self.hop_length)
        
        features['tempo'] = float(tempo)
        features['beat_times'] = beat_times
        features['num_beats'] = len(beats)
        
        # 2. Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        onset_times = librosa.times_like(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        features['onset_times'] = onset_times
        features['num_onsets'] = len(onset_frames)
        
        # 3. Rhythm patterns (tempo grams)
        tempogram = librosa.feature.tempogram(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        features['tempogram'] = tempogram
        
        # 4. Beat synchronous features
        if len(beats) > 1:
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=self.sr, hop_length=self.hop_length
            )
            # Synchronize to beats
            chroma_sync = librosa.util.sync(chroma, beats)
            features['chroma_sync'] = chroma_sync
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sr, hop_length=self.hop_length
            )
            # Synchronize to beats
            spectral_sync = librosa.util.sync(spectral_centroid, beats)
            features['spectral_sync'] = spectral_sync
        
        # 5. Rhythmic regularity
        if len(beat_times) > 2:
            beat_intervals = np.diff(beat_times)
            features['beat_interval_mean'] = np.mean(beat_intervals)
            features['beat_interval_std'] = np.std(beat_intervals)
            features['rhythmic_regularity'] = 1.0 / (1.0 + features['beat_interval_std'])
        
        # 6. Groove characteristics
        if len(onset_times) > 0 and len(beat_times) > 0:
            # Onset density per beat
            features['onset_density'] = len(onset_times) / len(beat_times)
            
            # Syncopation measure (simplified)
            syncopation_score = self._calculate_syncopation(onset_times, beat_times)
            features['syncopation'] = syncopation_score
        
        return features
    
    def _calculate_syncopation(self, onset_times: np.ndarray, 
                              beat_times: np.ndarray) -> float:
        """Calculate a simple syncopation measure."""
        if len(onset_times) == 0 or len(beat_times) < 2:
            return 0.0
        
        # Find onsets that are off-beat
        beat_tolerance = (beat_times[1] - beat_times[0]) * 0.2  # 20% tolerance
        
        off_beat_onsets = 0
        for onset in onset_times:
            # Check if onset is close to any beat
            distances_to_beats = np.abs(beat_times - onset)
            min_distance = np.min(distances_to_beats)
            
            if min_distance > beat_tolerance:
                off_beat_onsets += 1
        
        syncopation = off_beat_onsets / len(onset_times)
        return syncopation
    
    def compare_rhythm_preservation(self, original_audio: np.ndarray, 
                                  transferred_audio: np.ndarray) -> Dict[str, float]:
        """Compare rhythmic characteristics between original and transferred audio."""
        orig_features = self.extract_rhythmic_features(original_audio)
        trans_features = self.extract_rhythmic_features(transferred_audio)
        
        comparison = {}
        
        # 1. Tempo preservation
        tempo_diff = abs(orig_features['tempo'] - trans_features['tempo'])
        tempo_preservation = max(0, 1 - (tempo_diff / 50))  # Normalize by 50 BPM
        comparison['tempo_preservation'] = tempo_preservation
        
        # 2. Beat alignment (if both have beats)
        if (orig_features.get('beat_times') is not None and 
            trans_features.get('beat_times') is not None):
            
            # Use mir_eval if available for better beat evaluation
            if mir_eval is not None:
                try:
                    # Align beat sequences and calculate F-measure
                    f_measure = mir_eval.beat.f_measure(
                        orig_features['beat_times'][:min(len(orig_features['beat_times']), 100)],
                        trans_features['beat_times'][:min(len(trans_features['beat_times']), 100)]
                    )
                    comparison['beat_alignment'] = f_measure
                except:
                    # Fallback to simple correlation
                    comparison['beat_alignment'] = self._simple_beat_correlation(
                        orig_features['beat_times'], trans_features['beat_times']
                    )
            else:
                comparison['beat_alignment'] = self._simple_beat_correlation(
                    orig_features['beat_times'], trans_features['beat_times']
                )
        
        # 3. Rhythmic regularity preservation
        if ('rhythmic_regularity' in orig_features and 
            'rhythmic_regularity' in trans_features):
            regularity_diff = abs(
                orig_features['rhythmic_regularity'] - 
                trans_features['rhythmic_regularity']
            )
            comparison['regularity_preservation'] = max(0, 1 - regularity_diff)
        
        # 4. Syncopation preservation
        if ('syncopation' in orig_features and 'syncopation' in trans_features):
            syncopation_diff = abs(
                orig_features['syncopation'] - trans_features['syncopation']
            )
            comparison['syncopation_preservation'] = max(0, 1 - syncopation_diff)
        
        # 5. Overall rhythm preservation score
        scores = [v for v in comparison.values() if isinstance(v, (int, float))]
        comparison['overall_rhythm_preservation'] = np.mean(scores) if scores else 0.0
        
        return comparison
    
    def _simple_beat_correlation(self, beats1: np.ndarray, beats2: np.ndarray) -> float:
        """Simple beat correlation for when mir_eval is not available."""
        if len(beats1) < 2 or len(beats2) < 2:
            return 0.0
        
        # Create beat emphasis patterns
        duration = max(beats1[-1], beats2[-1])
        time_grid = np.arange(0, duration, 0.01)  # 10ms resolution
        
        emphasis1 = np.zeros_like(time_grid)
        emphasis2 = np.zeros_like(time_grid)
        
        # Mark beat positions
        for beat in beats1:
            idx = int(beat * 100)  # Convert to 10ms index
            if idx < len(emphasis1):
                emphasis1[idx] = 1.0
        
        for beat in beats2:
            idx = int(beat * 100)
            if idx < len(emphasis2):
                emphasis2[idx] = 1.0
        
        # Calculate correlation
        correlation = np.corrcoef(emphasis1, emphasis2)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0

class LyricalIntelligibilityAnalyzer:
    """
    Analyzes lyrical intelligibility in style-transferred vocal music.
    
    Uses speech recognition, phonetic analysis, and vocal clarity metrics
    to assess whether lyrics remain intelligible after style transfer.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        # Handle speech recognition module import safely
        try:
            import speech_recognition as sr_module
            self.recognizer = sr_module.Recognizer()
        except ImportError:
            self.recognizer = None
    
    def extract_vocal_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract vocal and speech-related features."""
        features = {}
        
        # 1. Vocal separation (simplified)
        vocal_audio = self._separate_vocals(audio)
        features['vocal_audio'] = vocal_audio
        
        # 2. Spectral features for vocal analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=vocal_audio, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=vocal_audio, sr=self.sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=vocal_audio, sr=self.sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(vocal_audio)
        
        features['spectral_centroid'] = np.mean(spectral_centroid)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        features['zero_crossing_rate'] = np.mean(zero_crossing_rate)
        
        # 3. MFCCs for phonetic content
        mfccs = librosa.feature.mfcc(y=vocal_audio, sr=self.sr, n_mfcc=13)
        features['mfccs'] = mfccs
        features['mfcc_stats'] = {
            'mean': np.mean(mfccs, axis=1),
            'std': np.std(mfccs, axis=1),
            'delta_mean': np.mean(np.diff(mfccs, axis=1), axis=1)
        }
        
        # 4. Vocal clarity metrics
        features['vocal_clarity'] = self._calculate_vocal_clarity(vocal_audio)
        
        # 5. Pitch contour for intonation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            vocal_audio, fmin=80, fmax=400, sr=self.sr
        )
        features['f0'] = f0
        features['voiced_frames'] = np.sum(voiced_flag)
        features['pitch_stability'] = np.std(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        
        return features
    
    def _separate_vocals(self, audio: np.ndarray) -> np.ndarray:
        """Simple vocal separation using harmonic-percussive decomposition."""
        # Use harmonic component as vocal approximation
        harmonic, _ = librosa.effects.hpss(audio)
        
        # Apply spectral filtering to enhance vocal frequencies
        stft = librosa.stft(harmonic)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Enhance vocal frequency range (roughly 80-1000 Hz)
        freqs = librosa.fft_frequencies(sr=self.sr)
        vocal_mask = (freqs >= 80) & (freqs <= 1000)
        
        # Create a gentle enhancement filter
        enhancement = np.ones_like(magnitude)
        enhancement[vocal_mask] *= 1.5
        
        enhanced_magnitude = magnitude * enhancement
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        
        vocal_audio = librosa.istft(enhanced_stft)
        return vocal_audio
    
    def _calculate_vocal_clarity(self, vocal_audio: np.ndarray) -> float:
        """Calculate a vocal clarity score based on spectral and temporal features."""
        # Harmonic-to-noise ratio approximation
        harmonic, percussive = librosa.effects.hpss(vocal_audio)
        
        harmonic_energy = np.mean(harmonic ** 2)
        percussive_energy = np.mean(percussive ** 2)
        
        if percussive_energy > 0:
            hnr = harmonic_energy / percussive_energy
            clarity_score = min(1.0, hnr / 10.0)  # Normalize to 0-1
        else:
            clarity_score = 1.0
        
        return clarity_score
    
    def recognize_speech(self, audio: np.ndarray, language: str = "en-US") -> Dict[str, Any]:
        """Attempt speech recognition on vocal audio."""
        if not self.recognizer:
            return {"error": "Speech recognition not available"}
        
        try:
            # Convert audio to format expected by speech_recognition
            import io
            import wave
            import tempfile
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Convert to int16 format
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # Write WAV file
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sr)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Recognize speech
                with sr.AudioFile(tmp_file.name) as source:
                    audio_data = self.recognizer.record(source)
                
                try:
                    text = self.recognizer.recognize_google(audio_data, language=language)
                    confidence = 1.0  # Google API doesn't return confidence
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                    
                    return {
                        "recognized_text": text,
                        "confidence": confidence,
                        "word_count": len(text.split()),
                        "intelligible": True
                    }
                
                except sr.UnknownValueError:
                    os.unlink(tmp_file.name)
                    return {
                        "recognized_text": "",
                        "confidence": 0.0,
                        "word_count": 0,
                        "intelligible": False
                    }
        
        except Exception as e:
            return {"error": f"Speech recognition failed: {str(e)}"}
    
    def compare_lyrical_intelligibility(self, original_audio: np.ndarray,
                                      transferred_audio: np.ndarray,
                                      language: str = "en-US") -> Dict[str, Any]:
        """Compare lyrical intelligibility between original and transferred audio."""
        orig_features = self.extract_vocal_features(original_audio)
        trans_features = self.extract_vocal_features(transferred_audio)
        
        comparison = {}
        
        # 1. Vocal clarity preservation
        clarity_orig = orig_features['vocal_clarity']
        clarity_trans = trans_features['vocal_clarity']
        clarity_preservation = min(clarity_trans / clarity_orig, 1.0) if clarity_orig > 0 else 0.0
        comparison['vocal_clarity_preservation'] = clarity_preservation
        
        # 2. Spectral feature preservation
        spectral_features = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']
        
        for feature in spectral_features:
            orig_val = orig_features[feature]
            trans_val = trans_features[feature]
            
            if orig_val > 0:
                preservation = 1.0 - abs(orig_val - trans_val) / orig_val
                comparison[f'{feature}_preservation'] = max(0, preservation)
        
        # 3. MFCC similarity (phonetic content)
        orig_mfcc_mean = orig_features['mfcc_stats']['mean']
        trans_mfcc_mean = trans_features['mfcc_stats']['mean']
        
        mfcc_correlation = np.corrcoef(orig_mfcc_mean, trans_mfcc_mean)[0, 1]
        comparison['phonetic_preservation'] = max(0, mfcc_correlation) if not np.isnan(mfcc_correlation) else 0.0
        
        # 4. Pitch stability preservation
        orig_stability = orig_features['pitch_stability']
        trans_stability = trans_features['pitch_stability']
        
        stability_preservation = min(orig_stability / trans_stability, 1.0) if trans_stability > 0 else 0.0
        comparison['pitch_stability_preservation'] = stability_preservation
        
        # 5. Speech recognition comparison (if available)
        if self.recognizer:
            orig_speech = self.recognize_speech(orig_features['vocal_audio'], language)
            trans_speech = self.recognize_speech(trans_features['vocal_audio'], language)
            
            comparison['original_recognition'] = orig_speech
            comparison['transferred_recognition'] = trans_speech
            
            # Calculate word similarity (simplified)
            if (orig_speech.get('intelligible') and trans_speech.get('intelligible')):
                orig_words = set(orig_speech['recognized_text'].lower().split())
                trans_words = set(trans_speech['recognized_text'].lower().split())
                
                if orig_words:
                    word_overlap = len(orig_words.intersection(trans_words)) / len(orig_words)
                    comparison['word_recognition_similarity'] = word_overlap
        
        # 6. Overall intelligibility score
        scores = [v for k, v in comparison.items() 
                 if isinstance(v, (int, float)) and 'preservation' in k]
        comparison['overall_intelligibility'] = np.mean(scores) if scores else 0.0
        
        return comparison

class MusicalCoherenceAnalyzer:
    """
    Analyzes musical coherence and structure preservation in style transfers.
    
    Evaluates harmonic progression, musical form, key stability, and overall
    musical logic preservation.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
    
    def extract_harmonic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract harmonic and tonal features."""
        features = {}
        
        # 1. Chroma features for harmonic content
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, hop_length=self.hop_length)
        features['chroma'] = chroma
        features['chroma_stats'] = {
            'mean': np.mean(chroma, axis=1),
            'std': np.std(chroma, axis=1)
        }
        
        # 2. Tonnetz (harmonic network) features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sr)
        features['tonnetz'] = tonnetz
        features['tonnetz_stats'] = {
            'mean': np.mean(tonnetz, axis=1),
            'std': np.std(tonnetz, axis=1)
        }
        
        # 3. Key estimation
        chroma_mean = np.mean(chroma, axis=1)
        # Simple key estimation based on chroma profile correlation
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # C minor
        
        correlations_major = []
        correlations_minor = []
        
        for shift in range(12):
            major_shifted = np.roll(major_profile, shift)
            minor_shifted = np.roll(minor_profile, shift)
            
            corr_major = np.corrcoef(chroma_mean, major_shifted)[0, 1]
            corr_minor = np.corrcoef(chroma_mean, minor_shifted)[0, 1]
            
            correlations_major.append(corr_major if not np.isnan(corr_major) else 0)
            correlations_minor.append(corr_minor if not np.isnan(corr_minor) else 0)
        
        best_major = np.argmax(correlations_major)
        best_minor = np.argmax(correlations_minor)
        
        if correlations_major[best_major] > correlations_minor[best_minor]:
            features['estimated_key'] = f"{best_major}_major"
            features['key_confidence'] = correlations_major[best_major]
        else:
            features['estimated_key'] = f"{best_minor}_minor"
            features['key_confidence'] = correlations_minor[best_minor]
        
        # 4. Harmonic change rate
        chroma_delta = np.diff(chroma, axis=1)
        harmonic_change_rate = np.mean(np.sum(np.abs(chroma_delta), axis=0))
        features['harmonic_change_rate'] = harmonic_change_rate
        
        # 5. Consonance/dissonance estimation
        consonance_score = self._calculate_consonance(chroma_mean)
        features['consonance'] = consonance_score
        
        return features
    
    def _calculate_consonance(self, chroma: np.ndarray) -> float:
        """Calculate a simple consonance score based on chroma distribution."""
        # Weights for consonant intervals (simplified)
        consonance_weights = np.array([1.0, 0.2, 0.8, 0.3, 0.9, 0.7, 0.1, 1.0, 0.3, 0.8, 0.2, 0.9])
        
        # Normalize chroma
        chroma_norm = chroma / (np.sum(chroma) + 1e-8)
        
        # Calculate weighted consonance
        consonance = np.sum(chroma_norm * consonance_weights)
        return consonance
    
    def analyze_musical_structure(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze musical structure and form."""
        features = {}
        
        # 1. Tempo and beat structure
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        features['tempo'] = float(tempo)
        features['num_beats'] = len(beats)
        
        # 2. Structural segmentation using self-similarity
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        
        # Compute self-similarity matrix
        similarity_matrix = np.dot(chroma.T, chroma)
        similarity_matrix = similarity_matrix / (np.linalg.norm(chroma, axis=0, keepdims=True).T + 1e-8)
        similarity_matrix = similarity_matrix / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
        
        features['self_similarity'] = similarity_matrix
        
        # 3. Novelty curve for structure detection
        novelty_curve = np.mean(np.diff(similarity_matrix, axis=1), axis=0)
        features['novelty_curve'] = novelty_curve
        
        # 4. Estimate number of sections
        if len(novelty_curve) > 0:
            peaks = librosa.util.peak_pick(
                novelty_curve, 
                pre_max=3, 
                post_max=3, 
                pre_avg=3, 
                post_avg=3, 
                delta=0.1, 
                wait=3
            )
            features['num_sections'] = len(peaks) + 1  # +1 for the last section
        else:
            features['num_sections'] = 1
        
        return features
    
    def compare_musical_coherence(self, original_audio: np.ndarray,
                                transferred_audio: np.ndarray) -> Dict[str, Any]:
        """Compare musical coherence between original and transferred audio."""
        orig_harmonic = self.extract_harmonic_features(original_audio)
        trans_harmonic = self.extract_harmonic_features(transferred_audio)
        
        orig_structure = self.analyze_musical_structure(original_audio)
        trans_structure = self.analyze_musical_structure(transferred_audio)
        
        comparison = {}
        
        # 1. Key preservation
        orig_key_conf = orig_harmonic['key_confidence']
        trans_key_conf = trans_harmonic['key_confidence']
        
        if (orig_harmonic['estimated_key'].split('_')[0] == 
            trans_harmonic['estimated_key'].split('_')[0]):
            key_preservation = min(trans_key_conf / orig_key_conf, 1.0) if orig_key_conf > 0 else 0.5
        else:
            key_preservation = 0.2  # Different key, but might be intentional
        
        comparison['key_preservation'] = key_preservation
        
        # 2. Harmonic similarity
        orig_chroma_mean = orig_harmonic['chroma_stats']['mean']
        trans_chroma_mean = trans_harmonic['chroma_stats']['mean']
        
        harmonic_correlation = np.corrcoef(orig_chroma_mean, trans_chroma_mean)[0, 1]
        comparison['harmonic_similarity'] = max(0, harmonic_correlation) if not np.isnan(harmonic_correlation) else 0.0
        
        # 3. Consonance preservation
        consonance_diff = abs(orig_harmonic['consonance'] - trans_harmonic['consonance'])
        comparison['consonance_preservation'] = max(0, 1 - consonance_diff)
        
        # 4. Structural similarity
        tempo_diff = abs(orig_structure['tempo'] - trans_structure['tempo'])
        tempo_preservation = max(0, 1 - (tempo_diff / 50))  # Normalize by 50 BPM
        comparison['tempo_preservation'] = tempo_preservation
        
        # 5. Harmonic change rate similarity
        orig_change_rate = orig_harmonic['harmonic_change_rate']
        trans_change_rate = trans_harmonic['harmonic_change_rate']
        
        if orig_change_rate > 0:
            change_rate_preservation = min(trans_change_rate / orig_change_rate, 1.0)
        else:
            change_rate_preservation = 1.0 if trans_change_rate == 0 else 0.0
        
        comparison['harmonic_rhythm_preservation'] = change_rate_preservation
        
        # 6. Overall musical coherence
        coherence_scores = [
            comparison['key_preservation'],
            comparison['harmonic_similarity'],
            comparison['consonance_preservation'],
            comparison['tempo_preservation'],
            comparison['harmonic_rhythm_preservation']
        ]
        
        comparison['overall_musical_coherence'] = np.mean(coherence_scores)
        
        return comparison

class MusicalEvaluationSystem:
    """
    Comprehensive musical evaluation system combining all musical analysis components.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.rhythm_analyzer = RhythmPreservationAnalyzer(sr)
        self.lyrical_analyzer = LyricalIntelligibilityAnalyzer(sr)
        self.coherence_analyzer = MusicalCoherenceAnalyzer(sr)
    
    def evaluate_musical_quality(self, original_audio: np.ndarray,
                                transferred_audio: np.ndarray,
                                language: str = "en-US") -> Dict[str, Any]:
        """Comprehensive musical quality evaluation."""
        print("Running comprehensive musical evaluation...")
        
        results = {
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'audio_duration': len(transferred_audio) / self.sr,
            'sample_rate': self.sr
        }
        
        # 1. Rhythm preservation analysis
        print("  - Analyzing rhythm preservation...")
        rhythm_results = self.rhythm_analyzer.compare_rhythm_preservation(
            original_audio, transferred_audio
        )
        results['rhythm_preservation'] = rhythm_results
        
        # 2. Lyrical intelligibility analysis
        print("  - Analyzing lyrical intelligibility...")
        lyrical_results = self.lyrical_analyzer.compare_lyrical_intelligibility(
            original_audio, transferred_audio, language
        )
        results['lyrical_intelligibility'] = lyrical_results
        
        # 3. Musical coherence analysis
        print("  - Analyzing musical coherence...")
        coherence_results = self.coherence_analyzer.compare_musical_coherence(
            original_audio, transferred_audio
        )
        results['musical_coherence'] = coherence_results
        
        # 4. Overall musical quality score
        rhythm_score = rhythm_results.get('overall_rhythm_preservation', 0.0)
        lyrical_score = lyrical_results.get('overall_intelligibility', 0.0)
        coherence_score = coherence_results.get('overall_musical_coherence', 0.0)
        
        overall_score = np.mean([rhythm_score, lyrical_score, coherence_score])
        results['overall_musical_quality'] = overall_score
        
        # 5. Detailed breakdown
        results['score_breakdown'] = {
            'rhythm_preservation': rhythm_score,
            'lyrical_intelligibility': lyrical_score,
            'musical_coherence': coherence_score,
            'weighted_average': overall_score
        }
        
        print(f"  ✅ Musical evaluation complete. Overall score: {overall_score:.3f}")
        
        return results

def create_demo_musical_evaluation():
    """Create a demonstration of the musical evaluation system."""
    print("Creating Demo Musical Evaluation")
    print("=" * 40)
    
    # Create synthetic audio data for demonstration
    sr = 22050
    duration = 10.0  # 10 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Original audio: simple melody with rhythm
    freq_base = 220  # A3
    original_audio = (
        np.sin(2 * np.pi * freq_base * t) * 0.3 +
        np.sin(2 * np.pi * freq_base * 1.5 * t) * 0.2 +  # Fifth
        np.sin(2 * np.pi * freq_base * 2 * t) * 0.1       # Octave
    )
    
    # Add rhythm by modulating amplitude
    beat_freq = 2.0  # 2 beats per second (120 BPM)
    rhythm_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
    original_audio *= rhythm_envelope
    
    # Transferred audio: same melody with different timbre and slight tempo change
    freq_transferred = freq_base * 1.05  # Slightly higher pitch
    transferred_audio = (
        np.sin(2 * np.pi * freq_transferred * t) * 0.25 +
        np.sin(2 * np.pi * freq_transferred * 1.25 * t) * 0.25 +  # Different interval
        np.sin(2 * np.pi * freq_transferred * 1.75 * t) * 0.15 +  # Different harmonic
        np.random.normal(0, 0.05, len(t))  # Add some noise for realism
    )
    
    # Slightly different rhythm
    beat_freq_trans = 2.1  # Slightly faster
    rhythm_envelope_trans = 0.6 + 0.4 * np.sin(2 * np.pi * beat_freq_trans * t)
    transferred_audio *= rhythm_envelope_trans
    
    print(f"Generated synthetic audio:")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Original frequency: {freq_base} Hz")
    print(f"  Transferred frequency: {freq_transferred} Hz")
    
    # Create evaluation system
    evaluator = MusicalEvaluationSystem(sr=sr)
    
    # Run evaluation
    print(f"\nRunning musical evaluation...")
    results = evaluator.evaluate_musical_quality(
        original_audio, transferred_audio, language="en-US"
    )
    
    # Display results
    print(f"\n📊 MUSICAL EVALUATION RESULTS")
    print(f"=" * 40)
    
    print(f"Overall Musical Quality: {results['overall_musical_quality']:.3f}")
    print(f"\nDetailed Scores:")
    for component, score in results['score_breakdown'].items():
        if component != 'weighted_average':
            print(f"  {component.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"\nRhythm Preservation Details:")
    rhythm_data = results['rhythm_preservation']
    for key, value in rhythm_data.items():
        if isinstance(value, (int, float)):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
    
    print(f"\nMusical Coherence Details:")
    coherence_data = results['musical_coherence']
    for key, value in coherence_data.items():
        if isinstance(value, (int, float)):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
    
    # Save results
    output_dir = "experiments/musical_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    results_file = os.path.join(output_dir, "demo_musical_evaluation.json")
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, np.floating):
                        json_results[key][k] = float(v)
                    else:
                        json_results[key][k] = v
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, np.floating):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"✅ Demo Musical Evaluation Complete!")
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    results = create_demo_musical_evaluation()
    
    print(f"\n🎵 Musical Evaluation System Ready! 🎵")
    print(f"   Components: Rhythm, Lyrics, Coherence")
    print(f"   Overall Score: {results['overall_musical_quality']:.3f}")