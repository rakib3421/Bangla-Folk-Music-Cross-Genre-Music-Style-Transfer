"""
Musical structure analysis module for cross-genre music style transfer.
Implements vocal/instrumental separation, chord progression extraction, and rhythm pattern identification.
"""

import os
import librosa
import numpy as np
import pandas as pd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class MusicalStructureAnalyzer:
    def __init__(self, sr=44100, hop_length=512):
        """
        Initialize musical structure analyzer.
        
        Args:
            sr (int): Sample rate
            hop_length (int): Hop length for analysis
        """
        self.sr = sr
        self.hop_length = hop_length
        
    def separate_vocals_instruments(self, y):
        """
        Separate vocals and instruments using harmonic-percussive separation
        and spectral subtraction techniques.
        
        Args:
            y (np.array): Audio time series
            
        Returns:
            dict: Separated audio components
        """
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Foreground-background separation (rough vocal separation)
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=self.sr)))
        S_filter = np.minimum(S_full, S_filter)
        
        margin_i, margin_v = 2, 10
        power = 2
        
        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)
        
        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)
        
        # Apply masks
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full
        
        # Convert back to time domain
        y_foreground = librosa.istft(S_foreground * phase)
        y_background = librosa.istft(S_background * phase)
        
        return {
            'vocals': y_foreground,
            'instruments': y_background,
            'harmonic': y_harmonic,
            'percussive': y_percussive,
            'vocal_mask': mask_v,
            'instrument_mask': mask_i
        }
    
    def extract_chord_progressions(self, y):
        """
        Extract chord progressions using chroma features and chord recognition.
        
        Args:
            y (np.array): Audio time series
            
        Returns:
            dict: Chord progression analysis
        """
        # Extract chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Beat-synchronous chroma
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        
        if len(beats) > 1:
            chroma_sync = librosa.util.sync(chroma, beats)
        else:
            chroma_sync = chroma
        
        # Simple chord recognition based on chroma patterns
        # This is a simplified approach - more sophisticated methods would use HMMs or neural networks
        chord_templates = self._get_chord_templates()
        
        # Compute similarity with chord templates
        chord_similarities = []
        for template in chord_templates:
            similarity = np.dot(chroma_sync.T, template)
            chord_similarities.append(similarity)
        
        chord_similarities = np.array(chord_similarities).T
        
        # Find most likely chord for each time frame
        chord_indices = np.argmax(chord_similarities, axis=1)
        chord_names = list(chord_templates.keys())
        estimated_chords = [chord_names[i] for i in chord_indices]
        
        # Analyze chord progression patterns
        chord_transitions = self._analyze_chord_transitions(estimated_chords)
        
        return {
            'chroma': chroma,
            'chroma_sync': chroma_sync,
            'beats': beats,
            'tempo': tempo,
            'estimated_chords': estimated_chords,
            'chord_similarities': chord_similarities,
            'chord_transitions': chord_transitions
        }
    
    def _get_chord_templates(self):
        """
        Get basic chord templates for chord recognition.
        
        Returns:
            dict: Chord templates
        """
        # Major chord templates (simplified)
        chord_templates = {}
        
        # Major chords
        major_pattern = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        minor_pattern = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i, note in enumerate(notes):
            # Major chord
            major_chord = np.roll(major_pattern, i)
            chord_templates[f'{note}:maj'] = major_chord / np.linalg.norm(major_chord)
            
            # Minor chord
            minor_chord = np.roll(minor_pattern, i)
            chord_templates[f'{note}:min'] = minor_chord / np.linalg.norm(minor_chord)
        
        return chord_templates
    
    def _analyze_chord_transitions(self, chords):
        """
        Analyze patterns in chord transitions.
        
        Args:
            chords (list): List of estimated chords
            
        Returns:
            dict: Chord transition analysis
        """
        if len(chords) < 2:
            return {'transitions': [], 'common_progressions': []}
        
        # Create transition pairs
        transitions = []
        for i in range(len(chords) - 1):
            transitions.append((chords[i], chords[i + 1]))
        
        # Count transition frequencies
        from collections import Counter
        transition_counts = Counter(transitions)
        
        # Find common progressions (sequences of 3+ chords)
        common_progressions = []
        for length in [3, 4]:
            if len(chords) >= length:
                progressions = []
                for i in range(len(chords) - length + 1):
                    prog = tuple(chords[i:i + length])
                    progressions.append(prog)
                
                prog_counts = Counter(progressions)
                # Keep progressions that appear more than once
                common = [(prog, count) for prog, count in prog_counts.items() if count > 1]
                common_progressions.extend(common)
        
        return {
            'transitions': transition_counts.most_common(10),
            'common_progressions': common_progressions
        }
    
    def identify_rhythm_patterns(self, y):
        """
        Identify rhythm patterns and meter.
        
        Args:
            y (np.array): Audio time series
            
        Returns:
            dict: Rhythm pattern analysis
        """
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        # Rhythm strength (using tempogram)
        tempogram = librosa.feature.tempogram(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Beat synchronous features for pattern analysis
        if len(beats) > 1:
            # Extract percussive component for rhythm analysis
            _, y_percussive = librosa.effects.hpss(y)
            
            # Beat-synchronous MFCC for rhythm timbre
            mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, hop_length=self.hop_length)
            mfcc_sync = librosa.util.sync(mfcc, beats)
            
            # Beat intervals for meter analysis
            beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
            beat_intervals = np.diff(beat_times)
            
            # Simple meter detection (duple vs triple)
            meter_strength = self._analyze_meter(beat_times, onset_times)
        else:
            mfcc_sync = np.array([])
            beat_intervals = np.array([])
            meter_strength = {'duple': 0, 'triple': 0}
        
        return {
            'tempo': tempo,
            'beats': beats,
            'onset_times': onset_times,
            'tempogram': tempogram,
            'beat_intervals': beat_intervals,
            'rhythm_mfcc': mfcc_sync,
            'meter_strength': meter_strength
        }
    
    def _analyze_meter(self, beat_times, onset_times, max_meter=4):
        """
        Analyze musical meter patterns.
        
        Args:
            beat_times (np.array): Beat time positions
            onset_times (np.array): Onset time positions
            max_meter (int): Maximum meter to analyze
            
        Returns:
            dict: Meter strength analysis
        """
        if len(beat_times) < max_meter:
            return {'duple': 0, 'triple': 0}
        
        meter_scores = {}
        
        # Analyze different meter patterns
        for meter in [2, 3, 4]:
            score = 0
            
            # Group beats by meter
            for i in range(0, len(beat_times) - meter + 1, meter):
                beat_group = beat_times[i:i + meter]
                
                # Count onsets that align with strong beats
                strong_beat_onsets = 0
                for j, beat_time in enumerate(beat_group):
                    # First beat is strongest, others are weaker
                    weight = 1.0 if j == 0 else 0.5
                    
                    # Find onsets close to this beat
                    close_onsets = np.sum(np.abs(onset_times - beat_time) < 0.1)
                    strong_beat_onsets += close_onsets * weight
                
                score += strong_beat_onsets
            
            meter_scores[f'meter_{meter}'] = score
        
        # Simplified duple vs triple classification
        duple_strength = meter_scores.get('meter_2', 0) + meter_scores.get('meter_4', 0)
        triple_strength = meter_scores.get('meter_3', 0)
        
        return {
            'duple': duple_strength,
            'triple': triple_strength,
            'details': meter_scores
        }
    
    def extract_melodic_contour(self, y):
        """
        Extract melodic contour from audio.
        
        Args:
            y (np.array): Audio time series
            
        Returns:
            dict: Melodic analysis
        """
        # Fundamental frequency estimation
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Remove unvoiced segments (f0 = 0)
        voiced_f0 = f0[f0 > 0]
        
        if len(voiced_f0) == 0:
            return {
                'f0': f0,
                'melodic_intervals': np.array([]),
                'pitch_range': 0,
                'pitch_stability': 0
            }
        
        # Convert to semitones for musical analysis
        f0_semitones = librosa.hz_to_midi(voiced_f0)
        
        # Melodic intervals
        melodic_intervals = np.diff(f0_semitones)
        
        # Pitch range
        pitch_range = np.max(f0_semitones) - np.min(f0_semitones)
        
        # Pitch stability (inverse of standard deviation)
        pitch_stability = 1.0 / (np.std(f0_semitones) + 1e-6)
        
        return {
            'f0': f0,
            'f0_semitones': f0_semitones,
            'melodic_intervals': melodic_intervals,
            'pitch_range': pitch_range,
            'pitch_stability': pitch_stability
        }
    
    def analyze_complete_structure(self, audio_file):
        """
        Perform complete musical structure analysis.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            dict: Complete structural analysis
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            analysis = {}
            
            # Vocal/instrumental separation
            separation = self.separate_vocals_instruments(y)
            analysis['separation'] = separation
            
            # Chord progression analysis
            chord_analysis = self.extract_chord_progressions(y)
            analysis['chords'] = chord_analysis
            
            # Rhythm pattern analysis
            rhythm_analysis = self.identify_rhythm_patterns(y)
            analysis['rhythm'] = rhythm_analysis
            
            # Melodic contour (on vocal/harmonic component)
            melodic_analysis = self.extract_melodic_contour(separation['vocals'])
            analysis['melody'] = melodic_analysis
            
            # File info
            analysis['file'] = audio_file
            analysis['duration'] = len(y) / sr
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing structure of {audio_file}: {str(e)}")
            return None

def analyze_genre_musical_structure(audio_files, genre_name, max_files=10):
    """
    Analyze musical structure characteristics of a genre.
    
    Args:
        audio_files (list): List of audio file paths
        genre_name (str): Name of the genre
        max_files (int): Maximum number of files to analyze
        
    Returns:
        dict: Genre structure analysis
    """
    analyzer = MusicalStructureAnalyzer()
    
    # Sample files for analysis
    sample_files = np.random.choice(audio_files, min(max_files, len(audio_files)), replace=False)
    
    print(f"Analyzing musical structure of {len(sample_files)} {genre_name} files...")
    
    analyses = []
    for audio_file in sample_files:
        analysis = analyzer.analyze_complete_structure(audio_file)
        if analysis:
            analyses.append(analysis)
    
    if not analyses:
        return None
    
    # Aggregate analysis results
    genre_characteristics = {
        'genre': genre_name,
        'sample_count': len(analyses),
        'tempo_stats': [],
        'chord_patterns': [],
        'rhythm_patterns': [],
        'melodic_characteristics': []
    }
    
    for analysis in analyses:
        # Collect tempo information
        if 'rhythm' in analysis and 'tempo' in analysis['rhythm']:
            genre_characteristics['tempo_stats'].append(analysis['rhythm']['tempo'])
        
        # Collect chord patterns
        if 'chords' in analysis and 'estimated_chords' in analysis['chords']:
            genre_characteristics['chord_patterns'].extend(analysis['chords']['estimated_chords'])
        
        # Collect melodic characteristics
        if 'melody' in analysis:
            melody = analysis['melody']
            genre_characteristics['melodic_characteristics'].append({
                'pitch_range': melody.get('pitch_range', 0),
                'pitch_stability': melody.get('pitch_stability', 0)
            })
    
    # Compute summary statistics
    if genre_characteristics['tempo_stats']:
        tempos = genre_characteristics['tempo_stats']
        genre_characteristics['tempo_summary'] = {
            'mean': np.mean(tempos),
            'std': np.std(tempos),
            'range': [np.min(tempos), np.max(tempos)]
        }
    
    return genre_characteristics

if __name__ == "__main__":
    # Example usage
    analyzer = MusicalStructureAnalyzer()
    
    # Example analysis on a single file
    # analysis = analyzer.analyze_complete_structure("path/to/audio/file.wav")
    print("Musical Structure Analyzer initialized successfully!")