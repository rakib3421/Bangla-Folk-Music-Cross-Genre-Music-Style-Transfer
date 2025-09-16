"""
Audio preprocessing utilities for cross-genre music style transfer.
Handles format standardization, normalization, and segmentation.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, target_sr=44100, target_format='wav', segment_duration=30):
        """
        Initialize audio preprocessor with target specifications.
        
        Args:
            target_sr (int): Target sample rate (44.1kHz)
            target_format (str): Target audio format
            segment_duration (int): Duration for segmentation in seconds
        """
        self.target_sr = target_sr
        self.target_format = target_format
        self.segment_duration = segment_duration
        
    def load_audio(self, file_path):
        """
        Load audio file for processing (required by Flask app).
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            print(f"DEBUG: load_audio called with file_path={file_path}")
            print(f"DEBUG: file exists: {os.path.exists(file_path) if isinstance(file_path, str) else 'not a string'}")
            
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            # Normalize audio
            y = librosa.util.normalize(y)
            print(f"DEBUG: Successfully loaded audio, shape={y.shape}, sr={sr}")
            return y, sr
        except Exception as e:
            print(f"Error loading audio {file_path}: {str(e)}")
            print(f"DEBUG: Exception type: {type(e)}")
            raise
    
    def save_audio(self, audio, output_path, sr):
        """
        Save audio to file (required by Flask app).
        
        Args:
            audio: Audio data (numpy array)
            output_path: Output file path
            sr: Sample rate
        """
        try:
            # Debug print to understand the parameters
            print(f"DEBUG: save_audio called with audio.shape={getattr(audio, 'shape', 'no shape')}, output_path={output_path}, sr={sr}")
            
            sf.write(output_path, audio, sr, subtype='PCM_16')
            print(f"Successfully saved audio to: {output_path}")
        except Exception as e:
            print(f"Error saving audio to '{output_path}' with sr={sr}: {str(e)}")
            # Try alternative method without subtype
            try:
                sf.write(output_path, audio, sr)
                print(f"Successfully saved audio using fallback method: {output_path}")
            except Exception as e2:
                print(f"Fallback save also failed: {str(e2)}")
                raise e
        
    def standardize_audio_file(self, input_path, output_path):
        """
        Convert audio file to standard format with normalization.
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to output standardized file
        """
        try:
            # Load audio file
            y, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
            
            # Normalize audio to prevent clipping
            y = librosa.util.normalize(y)
            
            # Save as 16-bit WAV
            sf.write(output_path, y, self.target_sr, subtype='PCM_16')
            
            return True, len(y) / self.target_sr  # Return success and duration
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            return False, 0
    
    def segment_audio(self, audio_path, output_dir, min_duration=10):
        """
        Segment long audio files into manageable chunks.
        
        Args:
            audio_path (str): Path to audio file
            output_dir (str): Directory to save segments
            min_duration (int): Minimum duration for a segment
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            duration = len(y) / sr
            
            # If file is shorter than segment duration, copy as is
            if duration <= self.segment_duration:
                base_name = Path(audio_path).stem
                output_path = os.path.join(output_dir, f"{base_name}_segment_01.wav")
                sf.write(output_path, y, sr, subtype='PCM_16')
                return [output_path]
            
            # Create segments
            segment_samples = self.segment_duration * sr
            segments = []
            
            for i, start in enumerate(range(0, len(y), segment_samples)):
                end = min(start + segment_samples, len(y))
                segment = y[start:end]
                
                # Skip segments that are too short
                if len(segment) / sr < min_duration:
                    continue
                
                base_name = Path(audio_path).stem
                output_path = os.path.join(output_dir, f"{base_name}_segment_{i+1:02d}.wav")
                sf.write(output_path, segment, sr, subtype='PCM_16')
                segments.append(output_path)
            
            return segments
            
        except Exception as e:
            print(f"Error segmenting {audio_path}: {str(e)}")
            return []
    
    def process_genre_dataset(self, input_dir, output_dir, segment=True):
        """
        Process all audio files in a genre directory.
        
        Args:
            input_dir (str): Input directory containing audio files
            output_dir (str): Output directory for processed files
            segment (bool): Whether to segment long files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f"*{ext}"))
        
        print(f"Found {len(audio_files)} audio files in {input_dir}")
        
        processed_files = []
        stats = {
            'total_files': len(audio_files),
            'successful': 0,
            'failed': 0,
            'total_duration': 0,
            'segments_created': 0
        }
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            if segment:
                # Create segments
                segments = self.segment_audio(str(audio_file), output_dir)
                if segments:
                    processed_files.extend(segments)
                    stats['segments_created'] += len(segments)
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
            else:
                # Direct conversion
                output_name = f"{audio_file.stem}.wav"
                output_path = os.path.join(output_dir, output_name)
                
                success, duration = self.standardize_audio_file(str(audio_file), output_path)
                if success:
                    processed_files.append(output_path)
                    stats['successful'] += 1
                    stats['total_duration'] += duration
                else:
                    stats['failed'] += 1
        
        return processed_files, stats
    
    def get_audio_info(self, file_path):
        """
        Get basic information about an audio file.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            dict: Audio file information
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            return {
                'file_path': file_path,
                'sample_rate': sr,
                'duration': duration,
                'num_samples': len(y),
                'file_size': os.path.getsize(file_path)
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e)
            }

def create_dataset_info(data_dir):
    """
    Create a comprehensive overview of the dataset.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        pd.DataFrame: Dataset information
    """
    preprocessor = AudioPreprocessor()
    dataset_info = []
    
    genres = ['Bangla Folk', 'Jazz', 'Rock']
    
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.exists(genre_dir):
            continue
            
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(genre_dir).glob(f"*{ext}"))
        
        print(f"Analyzing {len(audio_files)} files in {genre}...")
        
        for audio_file in tqdm(audio_files[:10], desc=f"Sampling {genre} files"):  # Sample first 10 files
            info = preprocessor.get_audio_info(str(audio_file))
            info['genre'] = genre
            dataset_info.append(info)
    
    return pd.DataFrame(dataset_info)

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    processed_dir = "data/processed"
    
    # Create dataset overview
    print("Creating dataset overview...")
    df = create_dataset_info(data_dir)
    print(df.groupby('genre').agg({
        'duration': ['count', 'mean', 'std', 'sum'],
        'sample_rate': 'nunique',
        'file_size': 'sum'
    }).round(2))