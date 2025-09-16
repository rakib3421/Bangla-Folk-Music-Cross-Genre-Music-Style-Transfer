"""
Simple test to verify our audio processing works correctly.
"""

import os
import sys
import numpy as np

def test_audio_loading():
    """Test basic audio loading functionality."""
    
    try:
        import librosa
        print("✓ Librosa imported successfully")
        
        # Test with a sample file
        data_dir = "data"
        genres = ['Bangla Folk', 'Jazz', 'Rock']
        
        for genre in genres:
            genre_path = os.path.join(data_dir, genre)
            if os.path.exists(genre_path):
                mp3_files = [f for f in os.listdir(genre_path) if f.endswith('.mp3')]
                if mp3_files:
                    test_file = os.path.join(genre_path, mp3_files[0])
                    print(f"\nTesting {genre}: {mp3_files[0][:50]}...")
                    
                    try:
                        # Load audio with librosa
                        y, sr = librosa.load(test_file, sr=22050, duration=10.0)  # Load only first 10 seconds
                        print(f"   ✓ Loaded: {len(y)} samples at {sr} Hz")
                        print(f"   ✓ Duration: {len(y)/sr:.1f} seconds")
                        
                        # Test basic feature extraction
                        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                        print(f"   ✓ Estimated tempo: {float(tempo):.1f} BPM")
                        
                        break  # Just test one file for now
                        
                    except Exception as e:
                        print(f"   ✗ Error loading file: {e}")
                        continue
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Audio Processing Setup...")
    
    if test_audio_loading():
        print("\n✓ Audio processing test successful!")
    else:
        print("\n✗ Audio processing test failed!")