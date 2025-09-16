#!/usr/bin/env python3
"""
Debug script to understand tensor dimension issues
"""

import torch
import torchaudio
import numpy as np
import librosa
from advanced_preprocessing import AdvancedAudioPreprocessor
from cpu_optimization import CPUOptimizedConfig

def debug_tensor_sizes():
    print("=== Debugging Tensor Sizes ===")
    
    # Initialize configuration
    config = CPUOptimizedConfig()
    preprocessor = AdvancedAudioPreprocessor(
        target_sr=config.audio_config['sample_rate'],
        n_mels=config.model_config['n_mels'],
        segment_length=config.audio_config['segment_length'],
        hop_length=config.audio_config['hop_length']
    )
    
    print(f"Config: SR={config.audio_config['sample_rate']}, n_mels={config.model_config['n_mels']}, segment_length={config.audio_config['segment_length']}")
    print(f"Segment samples: {preprocessor.segment_samples}")
    
    # Test with different audio files
    test_files = [
        "data/Bangla Folk/Aamar Hridoy.mp3",
        "data/Jazz/01 Maple Leaf Rag.mp3",
        "data/Rock/0 (1).mp3"
    ]
    
    for i, file_path in enumerate(test_files):
        try:
            print(f"\n--- Testing file {i+1}: {file_path.split('/')[-1]} ---")
            
            # Load audio
            audio = preprocessor.load_and_standardize_audio(file_path)
            print(f"Original audio length: {len(audio)} samples ({len(audio)/config.audio_config['sample_rate']:.2f}s)")
            
            # Extract mel spectrogram
            mel_spec = preprocessor.extract_mel_spectrogram(audio)
            print(f"Mel spectrogram shape: {mel_spec.shape}")
            print(f"Expected time frames: {preprocessor.segment_samples // preprocessor.hop_length + 1}")
            
            # Calculate expected dimensions
            expected_frames = preprocessor.segment_samples // preprocessor.hop_length + 1
            print(f"Expected shape: ({config.model_config['n_mels']}, {expected_frames})")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Test mel transform directly
    print(f"\n--- Testing Mel Transform Directly ---")
    
    # Create fixed-length audio
    test_audio = np.random.randn(preprocessor.segment_samples)
    print(f"Test audio length: {len(test_audio)} samples")
    
    # Apply mel transform
    mel_spec = preprocessor.extract_mel_spectrogram(test_audio)
    print(f"Result mel spec shape: {mel_spec.shape}")
    
    # Test hop length calculation
    expected_frames = len(test_audio) // preprocessor.hop_length + 1
    print(f"Expected frames from hop_length calculation: {expected_frames}")
    
    # Test with PyTorch MelSpectrogram directly
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.audio_config['sample_rate'],
        n_fft=preprocessor.n_fft,
        hop_length=preprocessor.hop_length,
        n_mels=config.model_config['n_mels'],
        f_min=80,
        f_max=config.audio_config['sample_rate'] // 2,
        power=2.0,
        normalized=True
    )
    
    audio_tensor = torch.FloatTensor(test_audio)
    mel_direct = mel_transform(audio_tensor)
    print(f"Direct PyTorch mel spec shape: {mel_direct.shape}")
    
    # Test different audio lengths
    print(f"\n--- Testing Different Audio Lengths ---")
    for length_sec in [2, 3, 4, 5]:
        length_samples = length_sec * config.audio_config['sample_rate']
        test_audio = np.random.randn(length_samples)
        
        # Apply preprocessing
        mel_spec = preprocessor.extract_mel_spectrogram(test_audio)
        print(f"{length_sec}s audio -> mel shape: {mel_spec.shape}")

if __name__ == "__main__":
    debug_tensor_sizes()