"""
Phase 5.2: Style Transfer Effectiveness Evaluation
===================================================

This module implements comprehensive evaluation metrics for measuring the effectiveness
of cross-genre music style transfer. Includes genre classification accuracy, spectral
similarity analysis, timbral feature comparison, and style transfer quality assessment.

Features:
- Genre classification accuracy using audio features
- Spectral similarity to target genre analysis
- Timbral feature extraction and comparison
- Style transfer quality scoring
- Cross-genre similarity metrics
- Batch evaluation for datasets
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pickle
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class TimbraleFeatureExtractor:
    """
    Comprehensive timbral feature extraction for music analysis.
    
    Extracts perceptually relevant timbral features including spectral,
    temporal, and harmonic characteristics for genre classification
    and style transfer evaluation.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize timbral feature extractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features from audio."""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        features = {}
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Spectral contrast (difference between peaks and valleys)
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=self.sample_rate)
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        features['spectral_contrast_std'] = float(np.std(spectral_contrast))
        
        # Spectral flatness (measure of noisiness)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        return features
    
    def extract_harmonic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract harmonic and tonal features."""
        features = {}
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        # Harmonic energy ratio
        harmonic_energy = np.sum(y_harmonic**2)
        total_energy = np.sum(audio**2)
        features['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-10))
        
        # Percussive energy ratio
        percussive_energy = np.sum(y_percussive**2)
        features['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-10))
        
        # Pitch estimation
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        
        # Extract fundamental frequency candidates
        f0_candidates = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                f0_candidates.append(pitch)
        
        if len(f0_candidates) > 0:
            features['pitch_mean'] = float(np.mean(f0_candidates))
            features['pitch_std'] = float(np.std(f0_candidates))
            features['pitch_range'] = float(np.max(f0_candidates) - np.min(f0_candidates))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        # Chroma deviation (how much the chroma vector deviates from uniform)
        chroma_deviation = np.std(np.mean(chroma, axis=1))
        features['chroma_deviation'] = float(chroma_deviation)
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal and rhythmic features."""
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
        
        # Onset rate (onsets per second)
        features['onset_rate'] = float(len(onset_times) / (len(audio) / self.sample_rate))
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = float(tempo)
        
        # Beat consistency
        if len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
            beat_intervals = np.diff(beat_times)
            features['beat_consistency'] = float(1.0 / (1.0 + np.std(beat_intervals)))
        else:
            features['beat_consistency'] = 0.0
        
        # Rhythm patterns using tempogram
        tempogram = librosa.feature.tempogram(y=audio, sr=self.sample_rate)
        features['tempogram_mean'] = float(np.mean(tempogram))
        features['tempogram_std'] = float(np.std(tempogram))
        
        return features
    
    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """Extract MFCC features."""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract all timbral features."""
        features = {}
        
        # Spectral features
        features.update(self.extract_spectral_features(audio))
        
        # Harmonic features
        features.update(self.extract_harmonic_features(audio))
        
        # Temporal features
        features.update(self.extract_temporal_features(audio))
        
        # MFCC features
        features.update(self.extract_mfcc_features(audio))
        
        return features

class GenreClassifier:
    """
    Machine learning-based genre classifier for style transfer evaluation.
    
    Uses timbral features to classify audio into genres and measure
    the effectiveness of style transfer operations.
    """
    
    def __init__(self, feature_extractor: TimbraleFeatureExtractor = None):
        """
        Initialize genre classifier.
        
        Args:
            feature_extractor: Timbral feature extractor instance
        """
        if feature_extractor is None:
            self.feature_extractor = TimbraleFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # Multiple classifiers for ensemble prediction
        self.classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.genre_labels = []
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(self, audio_list: List[np.ndarray], 
                        genre_labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels for training."""
        features_list = []
        
        print(f"Extracting features from {len(audio_list)} audio samples...")
        
        for i, audio in enumerate(audio_list):
            print(f"  Processing {i+1}/{len(audio_list)}: {genre_labels[i]}")
            
            # Extract features
            features = self.feature_extractor.extract_all_features(audio)
            
            # Store feature names (from first sample)
            if i == 0:
                self.feature_names = list(features.keys())
            
            # Convert to array
            feature_vector = [features[name] for name in self.feature_names]
            features_list.append(feature_vector)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        
        # Encode genre labels
        unique_genres = list(set(genre_labels))
        self.genre_labels = unique_genres
        y = np.array([unique_genres.index(genre) for genre in genre_labels])
        
        return X, y
    
    def train(self, audio_list: List[np.ndarray], genre_labels: List[str]):
        """Train the genre classifier."""
        print("Training Genre Classifier")
        print("=" * 30)
        
        # Prepare features
        X, y = self.prepare_features(audio_list, genre_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Genres: {self.genre_labels}")
        
        # Train classifiers
        for name, classifier in self.classifiers.items():
            print(f"\nTraining {name}...")
            classifier.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"  Test accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        print(f"\n✅ Genre classifier training complete!")
    
    def predict(self, audio: np.ndarray) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """Predict genre and confidence for a single audio sample."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio)
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Ensemble prediction
        predictions = {}
        probabilities = {}
        
        for name, classifier in self.classifiers.items():
            pred = classifier.predict(feature_vector_scaled)[0]
            predictions[name] = self.genre_labels[pred]
            
            # Get probabilities if available
            if hasattr(classifier, 'predict_proba'):
                proba = classifier.predict_proba(feature_vector_scaled)[0]
                probabilities[name] = {
                    self.genre_labels[i]: float(prob) 
                    for i, prob in enumerate(proba)
                }
        
        # Majority vote for final prediction
        pred_counts = {}
        for pred in predictions.values():
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        final_prediction = max(pred_counts, key=pred_counts.get)
        confidence = pred_counts[final_prediction] / len(predictions)
        
        return {
            'predicted_genre': final_prediction,
            'confidence': confidence,
            'individual_predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before saving")
        
        model_data = {
            'classifiers': self.classifiers,
            'scaler': self.scaler,
            'genre_labels': self.genre_labels,
            'feature_names': self.feature_names,
            'feature_extractor': self.feature_extractor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifiers = model_data['classifiers']
        self.scaler = model_data['scaler']
        self.genre_labels = model_data['genre_labels']
        self.feature_names = model_data['feature_names']
        self.feature_extractor = model_data['feature_extractor']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

class StyleTransferEvaluator:
    """
    Comprehensive style transfer effectiveness evaluator.
    
    Evaluates how well audio has been transferred from source to target genre
    using multiple metrics including classification accuracy and feature similarity.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize style transfer evaluator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.feature_extractor = TimbraleFeatureExtractor(sample_rate)
        self.genre_classifier = GenreClassifier(self.feature_extractor)
    
    def compute_spectral_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Compute spectral similarity between two audio signals."""
        # Compute mel spectrograms
        mel1 = librosa.feature.melspectrogram(y=audio1, sr=self.sample_rate)
        mel2 = librosa.feature.melspectrogram(y=audio2, sr=self.sample_rate)
        
        # Convert to log scale
        log_mel1 = np.log(mel1 + 1e-10)
        log_mel2 = np.log(mel2 + 1e-10)
        
        # Compute correlation between average spectra
        avg_mel1 = np.mean(log_mel1, axis=1)
        avg_mel2 = np.mean(log_mel2, axis=1)
        
        # Handle edge cases
        if np.std(avg_mel1) == 0 or np.std(avg_mel2) == 0:
            return 0.0
        
        correlation = np.corrcoef(avg_mel1, avg_mel2)[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            correlation = 0.0
        
        return float(correlation)
    
    def compute_timbral_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> Dict[str, float]:
        """Compute timbral similarity between two audio signals."""
        # Extract features for both signals
        features1 = self.feature_extractor.extract_all_features(audio1)
        features2 = self.feature_extractor.extract_all_features(audio2)
        
        similarities = {}
        
        # Compute similarity for each feature category
        feature_categories = {
            'spectral': ['spectral_centroid_mean', 'spectral_rolloff_mean', 'spectral_bandwidth_mean'],
            'harmonic': ['harmonic_ratio', 'percussive_ratio', 'pitch_mean'],
            'temporal': ['tempo', 'onset_rate', 'beat_consistency'],
            'timbral': ['spectral_contrast_mean', 'spectral_flatness_mean', 'zcr_mean']
        }
        
        for category, feature_list in feature_categories.items():
            category_similarities = []
            
            for feature in feature_list:
                if feature in features1 and feature in features2:
                    val1 = features1[feature]
                    val2 = features2[feature]
                    
                    # Normalize and compute similarity
                    if val1 != 0 or val2 != 0:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            norm_diff = abs(val1 - val2) / max_val
                            similarity = 1.0 - min(norm_diff, 1.0)
                        else:
                            similarity = 1.0
                    else:
                        similarity = 1.0
                    
                    category_similarities.append(similarity)
            
            if category_similarities:
                similarities[f'{category}_similarity'] = np.mean(category_similarities)
            else:
                similarities[f'{category}_similarity'] = 0.0
        
        # Overall timbral similarity
        similarities['overall_timbral_similarity'] = np.mean(list(similarities.values()))
        
        return similarities
    
    def evaluate_style_transfer(self, original: np.ndarray, transferred: np.ndarray,
                               target_genre: str, reference_target: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate style transfer effectiveness.
        
        Args:
            original: Original audio
            transferred: Style-transferred audio
            target_genre: Target genre name
            reference_target: Optional reference audio from target genre
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # 1. Genre classification accuracy
        if self.genre_classifier.is_trained:
            prediction = self.genre_classifier.predict(transferred)
            
            results['predicted_genre'] = prediction['predicted_genre']
            results['genre_classification_accuracy'] = float(
                prediction['predicted_genre'] == target_genre
            )
            results['genre_confidence'] = prediction['confidence']
            
            # Genre probability for target genre
            if target_genre in prediction['probabilities'].get('random_forest', {}):
                results['target_genre_probability'] = prediction['probabilities']['random_forest'][target_genre]
            else:
                results['target_genre_probability'] = 0.0
        
        # 2. Spectral similarity to reference
        if reference_target is not None:
            results['spectral_similarity_to_target'] = self.compute_spectral_similarity(
                transferred, reference_target
            )
            
            # Timbral similarity to reference
            timbral_sim = self.compute_timbral_similarity(transferred, reference_target)
            results.update(timbral_sim)
        
        # 3. Preservation vs Transfer trade-off
        results['spectral_similarity_to_original'] = self.compute_spectral_similarity(
            original, transferred
        )
        
        # Transfer effectiveness score
        if reference_target is not None:
            target_similarity = results.get('spectral_similarity_to_target', 0.0)
            original_similarity = results['spectral_similarity_to_original']
            
            # Good transfer: high similarity to target, moderate dissimilarity to original
            transfer_score = target_similarity * (1.0 - 0.5 * original_similarity)
            results['style_transfer_effectiveness'] = max(0.0, transfer_score)
        else:
            # Fallback: use genre classification accuracy
            results['style_transfer_effectiveness'] = results.get('genre_classification_accuracy', 0.0)
        
        return results
    
    def evaluate_batch(self, originals: List[np.ndarray], transferred: List[np.ndarray],
                      target_genres: List[str], reference_targets: List[np.ndarray] = None) -> Dict[str, List[float]]:
        """Evaluate a batch of style transfers."""
        batch_results = {}
        
        if reference_targets is None:
            reference_targets = [None] * len(originals)
        
        for i, (orig, trans, genre, ref) in enumerate(
            zip(originals, transferred, target_genres, reference_targets)
        ):
            print(f"Evaluating transfer {i+1}/{len(originals)}: -> {genre}")
            
            result = self.evaluate_style_transfer(orig, trans, genre, ref)
            
            for metric, value in result.items():
                if metric not in batch_results:
                    batch_results[metric] = []
                
                if isinstance(value, (int, float)):
                    batch_results[metric].append(value)
        
        return batch_results

def test_style_transfer_evaluation():
    """Test style transfer evaluation with synthetic data."""
    print("Testing Style Transfer Effectiveness Evaluation")
    print("=" * 50)
    
    # Create synthetic audio samples for different genres
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Synthetic genre samples
    folk_audio = create_folk_sample(t, sample_rate)
    jazz_audio = create_jazz_sample(t, sample_rate)
    rock_audio = create_rock_sample(t, sample_rate)
    
    # Create training data
    training_audio = [folk_audio, jazz_audio, rock_audio] * 5  # More samples for better training
    training_labels = ['Folk', 'Jazz', 'Rock'] * 5
    
    # Initialize evaluator
    evaluator = StyleTransferEvaluator(sample_rate)
    
    # Train genre classifier
    print("\n1. Training Genre Classifier...")
    evaluator.genre_classifier.train(training_audio, training_labels)
    
    # Test style transfer evaluation
    print("\n2. Evaluating Style Transfers...")
    
    # Simulate transferred audio (simple spectral morphing)
    folk_to_jazz = morph_audio(folk_audio, jazz_audio, 0.7)
    jazz_to_rock = morph_audio(jazz_audio, rock_audio, 0.6)
    
    # Evaluate transfers
    results = {}
    
    print("\nEvaluating Folk -> Jazz transfer:")
    results['folk_to_jazz'] = evaluator.evaluate_style_transfer(
        folk_audio, folk_to_jazz, 'Jazz', jazz_audio
    )
    
    print("Evaluating Jazz -> Rock transfer:")
    results['jazz_to_rock'] = evaluator.evaluate_style_transfer(
        jazz_audio, jazz_to_rock, 'Rock', rock_audio
    )
    
    # Print results
    print(f"\n{'='*50}")
    print("STYLE TRANSFER EVALUATION RESULTS")
    print(f"{'='*50}")
    
    for transfer_name, result in results.items():
        print(f"\n{transfer_name.replace('_', ' -> ').title()}:")
        print(f"  Predicted Genre: {result.get('predicted_genre', 'Unknown')}")
        print(f"  Classification Accuracy: {result.get('genre_classification_accuracy', 0.0):.3f}")
        print(f"  Target Genre Probability: {result.get('target_genre_probability', 0.0):.3f}")
        print(f"  Style Transfer Effectiveness: {result.get('style_transfer_effectiveness', 0.0):.3f}")
        print(f"  Spectral Similarity to Target: {result.get('spectral_similarity_to_target', 0.0):.3f}")
        print(f"  Overall Timbral Similarity: {result.get('overall_timbral_similarity', 0.0):.3f}")
    
    print(f"\n✅ Style Transfer Evaluation Testing Complete!")
    
    return results

def create_folk_sample(t, sr):
    """Create synthetic folk audio sample."""
    # Simple melody with folk characteristics
    melody = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)
    rhythm = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))  # Simple 4/4 rhythm
    return melody * rhythm * 0.3

def create_jazz_sample(t, sr):
    """Create synthetic jazz audio sample."""
    # Complex harmony with swing rhythm
    harmony = (np.sin(2 * np.pi * 220 * t) + 
               0.8 * np.sin(2 * np.pi * 275 * t) +  # Major third
               0.6 * np.sin(2 * np.pi * 330 * t) +  # Perfect fifth
               0.4 * np.sin(2 * np.pi * 392 * t))   # Seventh
    swing = 0.4 * (1 + np.sin(2 * np.pi * 2.5 * t + np.pi/3))
    return harmony * swing * 0.3

def create_rock_sample(t, sr):
    """Create synthetic rock audio sample."""
    # Power chord with driving rhythm
    power_chord = (np.sin(2 * np.pi * 220 * t) + 
                   0.9 * np.sin(2 * np.pi * 330 * t))  # Root and fifth
    drive = 0.6 * (1 + np.sin(2 * np.pi * 4 * t))  # Driving 4/4 beat
    distortion = np.tanh(power_chord * 2) * 0.5  # Simple distortion
    return distortion * drive

def morph_audio(source, target, alpha):
    """Simple spectral morphing between two audio signals."""
    # Compute STFTs
    source_stft = librosa.stft(source)
    target_stft = librosa.stft(target)
    
    # Ensure same size
    min_shape = (min(source_stft.shape[0], target_stft.shape[0]),
                 min(source_stft.shape[1], target_stft.shape[1]))
    
    source_stft = source_stft[:min_shape[0], :min_shape[1]]
    target_stft = target_stft[:min_shape[0], :min_shape[1]]
    
    # Morph magnitude, preserve source phase
    source_mag = np.abs(source_stft)
    target_mag = np.abs(target_stft)
    source_phase = np.angle(source_stft)
    
    # Interpolate magnitudes
    morphed_mag = (1 - alpha) * source_mag + alpha * target_mag
    
    # Combine with source phase
    morphed_stft = morphed_mag * np.exp(1j * source_phase)
    
    # Convert back to audio
    morphed_audio = librosa.istft(morphed_stft)
    
    return morphed_audio

if __name__ == "__main__":
    test_style_transfer_evaluation()