"""
Feature extraction module for cross-genre music style transfer.
Extracts mel-spectrograms, chromagrams, rhythm features, and timbral features.
"""

import os
import librosa
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sr=44100, hop_length=512, n_fft=2048):
        """
        Initialize feature extractor with audio processing parameters.
        
        Args:
            sr (int): Sample rate
            hop_length (int): Hop length for STFT
            n_fft (int): FFT window size
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        
    def extract_mel_spectrogram(self, y, n_mels=128):
        """
        Extract mel-spectrogram features.
        
        Args:
            y (np.array): Audio time series
            n_mels (int): Number of mel bands
            
        Returns:
            np.array: Mel-spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=n_mels,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chromagram(self, y, n_chroma=12):
        """
        Extract chromagram for harmonic content analysis.
        
        Args:
            y (np.array): Audio time series
            n_chroma (int): Number of chroma bins
            
        Returns:
            np.array: Chromagram
        """
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )
        return chroma
    
    def extract_rhythm_features(self, y):
        """
        Extract rhythm-related features including tempo and beat tracking.
        
        Args:
            y (np.array): Audio time series
            
        Returns:
            dict: Rhythm features
        """
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        
        # Rhythm patterns (tempogram)
        tempogram = librosa.feature.tempogram(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        # Beat synchronous chroma
        chroma_sync = librosa.feature.chroma_stft(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        if len(beats) > 0:
            chroma_sync = librosa.util.sync(chroma_sync, beats)
        
        return {
            'tempo': tempo,
            'beats': beats,
            'tempogram': tempogram,
            'beat_chroma': chroma_sync,
            'beat_count': len(beats)
        }
    
    def extract_timbral_features(self, y):
        """
        Extract timbral features (MFCC, spectral features).
        
        Args:
            y (np.array): Audio time series
            
        Returns:
            dict: Timbral features
        """
        # MFCC features
        mfccs = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=13,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        
        return {
            'mfcc': mfccs,
            'spectral_centroid': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zero_crossing_rate,
            'rms': rms
        }
    
    def extract_all_features(self, audio_file):
        """
        Extract all features from an audio file.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            dict: Complete feature set
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            # Extract features
            features = {}
            
            # Mel-spectrogram
            features['mel_spectrogram'] = self.extract_mel_spectrogram(y)
            
            # Chromagram
            features['chromagram'] = self.extract_chromagram(y)
            
            # Rhythm features
            rhythm_features = self.extract_rhythm_features(y)
            features.update(rhythm_features)
            
            # Timbral features
            timbral_features = self.extract_timbral_features(y)
            features.update(timbral_features)
            
            # Basic audio info
            features['duration'] = len(y) / sr
            features['audio_file'] = audio_file
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {str(e)}")
            return None
    
    def compute_feature_statistics(self, features):
        """
        Compute statistical summaries of time-varying features.
        
        Args:
            features (dict): Feature dictionary
            
        Returns:
            dict: Statistical summaries
        """
        stats_features = {}
        
        # Features to compute stats for
        time_varying_features = [
            'mel_spectrogram', 'chromagram', 'tempogram', 
            'mfcc', 'spectral_centroid', 'spectral_rolloff',
            'spectral_bandwidth', 'zero_crossing_rate', 'rms'
        ]
        
        for feature_name in time_varying_features:
            if feature_name in features:
                feature_data = features[feature_name]
                
                if feature_data.ndim > 1:
                    # For 2D features, compute stats across time axis
                    stats_features[f'{feature_name}_mean'] = np.mean(feature_data, axis=1)
                    stats_features[f'{feature_name}_std'] = np.std(feature_data, axis=1)
                    stats_features[f'{feature_name}_median'] = np.median(feature_data, axis=1)
                else:
                    # For 1D features
                    stats_features[f'{feature_name}_mean'] = np.mean(feature_data)
                    stats_features[f'{feature_name}_std'] = np.std(feature_data)
                    stats_features[f'{feature_name}_median'] = np.median(feature_data)
        
        # Add scalar features directly
        scalar_features = ['tempo', 'beat_count', 'duration']
        for feature_name in scalar_features:
            if feature_name in features:
                stats_features[feature_name] = features[feature_name]
        
        return stats_features

class GenreAnalyzer:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
    
    def analyze_genre_characteristics(self, audio_files, genre_name):
        """
        Analyze characteristic features of a specific genre.
        
        Args:
            audio_files (list): List of audio file paths
            genre_name (str): Name of the genre
            
        Returns:
            dict: Genre analysis results
        """
        print(f"Analyzing {len(audio_files)} {genre_name} files...")
        
        all_features = []
        
        # Extract features from a subset of files for analysis
        sample_size = min(20, len(audio_files))
        sample_files = np.random.choice(audio_files, sample_size, replace=False)
        
        for audio_file in sample_files:
            features = self.feature_extractor.extract_all_features(audio_file)
            if features:
                stats = self.feature_extractor.compute_feature_statistics(features)
                stats['genre'] = genre_name
                stats['file'] = audio_file
                all_features.append(stats)
        
        if not all_features:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_features)
        
        # Compute genre statistics
        genre_stats = {}
        
        # Tempo analysis
        if 'tempo' in df.columns:
            genre_stats['tempo'] = {
                'mean': df['tempo'].mean(),
                'std': df['tempo'].std(),
                'median': df['tempo'].median(),
                'range': [df['tempo'].min(), df['tempo'].max()]
            }
        
        # Spectral characteristics
        spectral_features = [col for col in df.columns if 'spectral' in col]
        for feature in spectral_features:
            if feature in df.columns:
                genre_stats[feature] = {
                    'mean': df[feature].mean() if df[feature].dtype in ['float64', 'int64'] else 'N/A',
                    'std': df[feature].std() if df[feature].dtype in ['float64', 'int64'] else 'N/A'
                }
        
        return {
            'genre': genre_name,
            'sample_count': len(all_features),
            'statistics': genre_stats,
            'raw_data': df
        }

def analyze_dataset_characteristics(data_dir):
    """
    Analyze characteristics of all genres in the dataset.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        dict: Complete dataset analysis
    """
    analyzer = GenreAnalyzer()
    results = {}
    
    genres = ['Bangla Folk', 'Jazz', 'Rock']
    
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.exists(genre_dir):
            continue
        
        # Get audio files
        from pathlib import Path
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(str(f) for f in Path(genre_dir).glob(f"*{ext}"))
        
        if audio_files:
            analysis = analyzer.analyze_genre_characteristics(audio_files, genre)
            if analysis:
                results[genre] = analysis
    
    return results

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    
    print("Analyzing dataset characteristics...")
    analysis_results = analyze_dataset_characteristics(data_dir)
    
    for genre, results in analysis_results.items():
        print(f"\n{genre} Analysis:")
        print(f"Sample size: {results['sample_count']}")
        
        if 'tempo' in results['statistics']:
            tempo_stats = results['statistics']['tempo']
            print(f"Tempo - Mean: {tempo_stats['mean']:.1f} BPM, Std: {tempo_stats['std']:.1f}")