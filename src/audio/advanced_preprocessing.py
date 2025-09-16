"""
Advanced Audio Preprocessing Pipeline for Cross-Genre Music Style Transfer
Implements comprehensive preprocessing with advanced augmentation techniques.
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Dict, List, Tuple, Optional, Union
import random
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioPreprocessor:
    """
    Advanced audio preprocessing with comprehensive augmentation capabilities.
    """
    
    def __init__(
        self,
        target_sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_length: float = 5.0,
        augmentation_prob: float = 0.7
    ):
        """
        Initialize the advanced preprocessor.
        
        Args:
            target_sr: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bins
            segment_length: Segment length in seconds
            augmentation_prob: Probability of applying augmentation
        """
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * target_sr)
        self.augmentation_prob = augmentation_prob
        
        # Initialize transform objects
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=80,
            f_max=target_sr // 2,
            power=2.0,
            normalized=True
        )
        
        # Time stretching factors
        self.time_stretch_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        # Pitch shift semitones
        self.pitch_shift_semitones = [-2, -1, 0, 1, 2]
        
        print(f"AdvancedAudioPreprocessor initialized:")
        print(f"  Target SR: {target_sr}")
        print(f"  Mel bins: {n_mels}")
        print(f"  Segment length: {segment_length}s")
        print(f"  Augmentation probability: {augmentation_prob}")
    
    def load_and_standardize_audio(self, file_path: str) -> np.ndarray:
        """
        Load and standardize audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Standardized audio array
        """
        try:
            # Try with librosa first (most reliable)
            audio, sr = librosa.load(file_path, sr=None)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize amplitude
            audio = librosa.util.normalize(audio)
            
            return audio
            
        except Exception as e:
            # Fallback to pydub for problematic files
            try:
                audio_segment = AudioSegment.from_file(file_path)
                
                # Convert to mono and resample
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(self.target_sr)
                
                # Normalize
                audio_segment = normalize(audio_segment)
                
                # Convert to numpy array
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
                
                return audio
                
            except Exception as e2:
                print(f"Failed to load {file_path}: {e2}")
                return None
    
    def apply_time_stretching(self, audio: np.ndarray, stretch_factor: float = None) -> np.ndarray:
        """
        Apply time stretching to audio.
        
        Args:
            audio: Input audio array
            stretch_factor: Time stretch factor (None for random)
            
        Returns:
            Time-stretched audio
        """
        if stretch_factor is None:
            stretch_factor = random.choice(self.time_stretch_factors)
        
        if stretch_factor == 1.0:
            return audio
        
        try:
            # Use librosa's time stretching
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            return stretched
        except Exception as e:
            print(f"Time stretching failed: {e}")
            return audio
    
    def apply_pitch_shifting(self, audio: np.ndarray, semitones: float = None) -> np.ndarray:
        """
        Apply pitch shifting to audio.
        
        Args:
            audio: Input audio array
            semitones: Number of semitones to shift (None for random)
            
        Returns:
            Pitch-shifted audio
        """
        if semitones is None:
            semitones = random.choice(self.pitch_shift_semitones)
        
        if semitones == 0:
            return audio
        
        try:
            # Use librosa's pitch shifting
            shifted = librosa.effects.pitch_shift(
                audio, 
                sr=self.target_sr, 
                n_steps=semitones
            )
            return shifted
        except Exception as e:
            print(f"Pitch shifting failed: {e}")
            return audio
    
    def add_background_noise(self, audio: np.ndarray, noise_level: float = None) -> np.ndarray:
        """
        Add background noise to audio.
        
        Args:
            audio: Input audio array
            noise_level: Noise amplitude (None for random)
            
        Returns:
            Audio with added noise
        """
        if noise_level is None:
            noise_level = random.uniform(0.005, 0.02)
        
        noise = np.random.normal(0, noise_level, audio.shape)
        noisy_audio = audio + noise
        
        # Ensure no clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        return noisy_audio
    
    def apply_frequency_masking(self, mel_spec: torch.Tensor, mask_param: int = 10) -> torch.Tensor:
        """
        Apply frequency masking to mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram tensor [mels, time]
            mask_param: Maximum mask width
            
        Returns:
            Frequency-masked mel spectrogram
        """
        n_mels, n_frames = mel_spec.shape
        
        # Random mask width
        mask_width = random.randint(1, min(mask_param, n_mels // 4))
        mask_start = random.randint(0, n_mels - mask_width)
        
        # Apply mask
        masked_spec = mel_spec.clone()
        masked_spec[mask_start:mask_start + mask_width, :] = mel_spec.min()
        
        return masked_spec
    
    def apply_time_masking(self, mel_spec: torch.Tensor, mask_param: int = 20) -> torch.Tensor:
        """
        Apply time masking to mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram tensor [mels, time]
            mask_param: Maximum mask width
            
        Returns:
            Time-masked mel spectrogram
        """
        n_mels, n_frames = mel_spec.shape
        
        # Random mask width
        mask_width = random.randint(1, min(mask_param, n_frames // 4))
        mask_start = random.randint(0, n_frames - mask_width)
        
        # Apply mask
        masked_spec = mel_spec.clone()
        masked_spec[:, mask_start:mask_start + mask_width] = mel_spec.min()
        
        return masked_spec
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract mel spectrogram from audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Mel spectrogram tensor
        """
        # Ensure fixed audio length for consistent mel-spectrogram size
        if len(audio) > self.segment_samples:
            # Trim to exact length
            audio = audio[:self.segment_samples]
        elif len(audio) < self.segment_samples:
            # Pad with zeros
            audio = np.pad(audio, (0, self.segment_samples - len(audio)), mode='constant')
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Convert to dB scale
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        return mel_spec_db.squeeze(0)  # Remove channel dimension
    
    def segment_audio(self, audio: np.ndarray, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            audio: Input audio array
            overlap: Overlap between segments (0.0 to 1.0)
            
        Returns:
            List of audio segments
        """
        if len(audio) <= self.segment_samples:
            # Pad if too short
            padded = np.pad(audio, (0, self.segment_samples - len(audio)), 'constant')
            return [padded]
        
        segments = []
        hop_samples = int(self.segment_samples * (1 - overlap))
        
        for start in range(0, len(audio) - self.segment_samples + 1, hop_samples):
            segment = audio[start:start + self.segment_samples]
            segments.append(segment)
        
        return segments
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Augmented audio
        """
        if random.random() > self.augmentation_prob:
            return audio
        
        augmented = audio.copy()
        
        # Time stretching (30% chance)
        if random.random() < 0.3:
            augmented = self.apply_time_stretching(augmented)
        
        # Pitch shifting (30% chance)
        if random.random() < 0.3:
            augmented = self.apply_pitch_shifting(augmented)
        
        # Background noise (20% chance)
        if random.random() < 0.2:
            augmented = self.add_background_noise(augmented)
        
        return augmented
    
    def augment_mel_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to mel spectrogram.
        
        Args:
            mel_spec: Input mel spectrogram
            
        Returns:
            Augmented mel spectrogram
        """
        if random.random() > self.augmentation_prob:
            return mel_spec
        
        augmented = mel_spec.clone()
        
        # Frequency masking (40% chance)
        if random.random() < 0.4:
            augmented = self.apply_frequency_masking(augmented)
        
        # Time masking (40% chance)
        if random.random() < 0.4:
            augmented = self.apply_time_masking(augmented)
        
        return augmented
    
    def process_audio_file(
        self, 
        file_path: str, 
        apply_augmentation: bool = True
    ) -> Dict[str, Union[List[torch.Tensor], Dict]]:
        """
        Complete processing pipeline for a single audio file.
        
        Args:
            file_path: Path to audio file
            apply_augmentation: Whether to apply augmentations
            
        Returns:
            Dictionary with processed segments and metadata
        """
        # Load and standardize audio
        audio = self.load_and_standardize_audio(file_path)
        if audio is None:
            return None
        
        # Apply audio-level augmentation
        if apply_augmentation:
            audio = self.augment_audio(audio)
        
        # Segment audio
        segments = self.segment_audio(audio)
        
        # Process each segment
        mel_spectrograms = []
        for segment in segments:
            # Extract mel spectrogram
            mel_spec = self.extract_mel_spectrogram(segment)
            
            # Apply spectrogram-level augmentation
            if apply_augmentation:
                mel_spec = self.augment_mel_spectrogram(mel_spec)
            
            mel_spectrograms.append(mel_spec)
        
        # Metadata
        metadata = {
            'file_path': file_path,
            'original_length': len(audio) / self.target_sr,
            'n_segments': len(segments),
            'mel_shape': mel_spectrograms[0].shape if mel_spectrograms else None,
            'augmented': apply_augmentation
        }
        
        return {
            'mel_spectrograms': mel_spectrograms,
            'metadata': metadata
        }
    
    def process_dataset(
        self, 
        data_dir: str, 
        output_dir: str = None,
        genres: List[str] = None,
        max_files_per_genre: int = None
    ) -> Dict[str, List]:
        """
        Process entire dataset with advanced preprocessing.
        
        Args:
            data_dir: Directory containing audio files
            output_dir: Directory to save processed data (optional)
            genres: List of genres to process
            max_files_per_genre: Maximum files per genre
            
        Returns:
            Dictionary of processed data by genre
        """
        if genres is None:
            genres = ['Bangla Folk', 'Jazz', 'Rock']
        
        processed_data = {genre: [] for genre in genres}
        stats = {genre: {'files': 0, 'segments': 0, 'failed': 0} for genre in genres}
        
        print(f"\nProcessing dataset from: {data_dir}")
        
        for genre in genres:
            genre_dir = os.path.join(data_dir, genre)
            if not os.path.exists(genre_dir):
                print(f"Warning: Genre directory not found: {genre_dir}")
                continue
            
            print(f"\nProcessing {genre}...")
            
            # Get audio files
            audio_files = []
            for ext in ['.mp3', '.wav', '.flac', '.m4a']:
                audio_files.extend(Path(genre_dir).glob(f'*{ext}'))
            
            # Limit files if specified
            if max_files_per_genre:
                audio_files = audio_files[:max_files_per_genre]
            
            for i, file_path in enumerate(audio_files):
                try:
                    print(f"  Processing {i+1}/{len(audio_files)}: {file_path.name}")
                    
                    # Process file
                    result = self.process_audio_file(str(file_path))
                    
                    if result is not None:
                        processed_data[genre].append(result)
                        stats[genre]['files'] += 1
                        stats[genre]['segments'] += result['metadata']['n_segments']
                    else:
                        stats[genre]['failed'] += 1
                        
                except Exception as e:
                    print(f"    Error: {e}")
                    stats[genre]['failed'] += 1
                    continue
        
        # Print statistics
        print(f"\n{'='*50}")
        print("PROCESSING STATISTICS")
        print(f"{'='*50}")
        
        total_files = 0
        total_segments = 0
        total_failed = 0
        
        for genre in genres:
            files = stats[genre]['files']
            segments = stats[genre]['segments']
            failed = stats[genre]['failed']
            
            print(f"{genre}:")
            print(f"  ✓ Files processed: {files}")
            print(f"  ✓ Segments created: {segments}")
            print(f"  ✗ Failed: {failed}")
            
            total_files += files
            total_segments += segments
            total_failed += failed
        
        print(f"\nTotal:")
        print(f"  Files: {total_files}")
        print(f"  Segments: {total_segments}")
        print(f"  Failed: {total_failed}")
        
        # Save processed data if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            for genre in genres:
                if processed_data[genre]:
                    output_file = os.path.join(output_dir, f"{genre.replace(' ', '_').lower()}_processed.pt")
                    torch.save(processed_data[genre], output_file)
                    print(f"Saved {genre} data to: {output_file}")
            
            # Save statistics
            stats_file = os.path.join(output_dir, "processing_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to: {stats_file}")
        
        return processed_data

def test_advanced_preprocessing():
    """Test the advanced preprocessing pipeline."""
    print("Testing Advanced Audio Preprocessing Pipeline...")
    
    try:
        # Initialize preprocessor
        preprocessor = AdvancedAudioPreprocessor(
            target_sr=22050,
            n_mels=128,
            segment_length=3.0,  # Shorter for testing
            augmentation_prob=0.8
        )
        
        # Test with small dataset
        print("\nTesting with sample files...")
        processed_data = preprocessor.process_dataset(
            data_dir="data",
            output_dir="processed_data_advanced",
            max_files_per_genre=2  # Very small for testing
        )
        
        # Test augmentation examples
        print("\nTesting augmentation techniques...")
        
        # Find a sample file
        sample_files = []
        for genre in ['Bangla Folk', 'Jazz', 'Rock']:
            genre_dir = f"data/{genre}"
            if os.path.exists(genre_dir):
                for ext in ['.mp3', '.wav']:
                    files = list(Path(genre_dir).glob(f'*{ext}'))
                    if files:
                        sample_files.append(str(files[0]))
                        break
                if sample_files:
                    break
        
        if sample_files:
            sample_file = sample_files[0]
            print(f"Using sample file: {Path(sample_file).name}")
            
            # Test different augmentations
            print("  Testing time stretching...")
            audio = preprocessor.load_and_standardize_audio(sample_file)
            if audio is not None:
                stretched = preprocessor.apply_time_stretching(audio, 1.1)
                print(f"    Original length: {len(audio)} samples")
                print(f"    Stretched length: {len(stretched)} samples")
            
            print("  Testing pitch shifting...")
            if audio is not None:
                shifted = preprocessor.apply_pitch_shifting(audio, 2)
                print(f"    Pitch shift applied successfully")
            
            print("  Testing noise addition...")
            if audio is not None:
                noisy = preprocessor.add_background_noise(audio, 0.01)
                noise_level = np.std(noisy - audio)
                print(f"    Added noise level: {noise_level:.6f}")
            
            print("  Testing spectrogram masking...")
            if audio is not None:
                mel_spec = preprocessor.extract_mel_spectrogram(audio)
                masked_freq = preprocessor.apply_frequency_masking(mel_spec)
                masked_time = preprocessor.apply_time_masking(mel_spec)
                print(f"    Mel spectrogram shape: {mel_spec.shape}")
                print(f"    Frequency masking applied")
                print(f"    Time masking applied")
        
        print(f"\n✓ Advanced preprocessing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Advanced preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_advanced_preprocessing()