"""
Demo script to test the Phase 1 implementation with a minimal example.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.getcwd())

try:
    from audio_preprocessing import AudioPreprocessor
    from feature_extraction import AudioFeatureExtractor
    from musical_structure_analysis import MusicalStructureAnalyzer
    print("✓ All modules imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of our modules."""
    
    print("\n" + "="*50)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*50)
    
    # Test 1: Check if data directories exist
    data_dir = "data"
    genres = ['Bangla Folk', 'Jazz', 'Rock']
    
    print("\n1. Checking dataset structure...")
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.exists(genre_path):
            file_count = len([f for f in os.listdir(genre_path) if f.endswith('.mp3')])
            print(f"   ✓ {genre}: {file_count} files found")
        else:
            print(f"   ✗ {genre}: Directory not found")
    
    # Test 2: Initialize processors
    print("\n2. Initializing processors...")
    try:
        preprocessor = AudioPreprocessor()
        print("   ✓ AudioPreprocessor initialized")
        
        feature_extractor = AudioFeatureExtractor()
        print("   ✓ AudioFeatureExtractor initialized")
        
        structure_analyzer = MusicalStructureAnalyzer()
        print("   ✓ MusicalStructureAnalyzer initialized")
        
    except Exception as e:
        print(f"   ✗ Processor initialization failed: {e}")
        return False
    
    # Test 3: Try to analyze one file from each genre
    print("\n3. Testing file analysis (first file from each genre)...")
    
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if not os.path.exists(genre_path):
            continue
            
        # Find first MP3 file
        mp3_files = [f for f in os.listdir(genre_path) if f.endswith('.mp3')]
        if not mp3_files:
            print(f"   ✗ {genre}: No MP3 files found")
            continue
        
        test_file = os.path.join(genre_path, mp3_files[0])
        print(f"   Testing {genre}: {mp3_files[0][:50]}...")
        
        try:
            # Test basic audio info
            info = preprocessor.get_audio_info(test_file)
            if 'error' not in info:
                print(f"     ✓ Duration: {info['duration']:.1f}s, SR: {info['sample_rate']} Hz")
            else:
                print(f"     ✗ Error: {info['error']}")
                
        except Exception as e:
            print(f"     ✗ Analysis failed: {e}")
    
    print("\n✓ Basic functionality test completed!")
    return True

def run_minimal_demo():
    """Run a minimal demo of the complete pipeline."""
    
    print("\n" + "="*50)
    print("RUNNING MINIMAL DEMO")
    print("="*50)
    
    try:
        from phase1_analysis import Phase1Pipeline
        
        # Create a demo pipeline with minimal processing
        pipeline = Phase1Pipeline(output_dir="demo_results")
        
        print("\n1. Running dataset overview...")
        dataset_stats = pipeline.step1_dataset_overview()
        
        print(f"\nDataset Summary:")
        total_files = sum(stats['file_count'] for stats in dataset_stats.values())
        total_size = sum(stats['size_mb'] for stats in dataset_stats.values())
        print(f"Total files: {total_files}")
        print(f"Total size: {total_size:.1f} MB")
        
        print("\n✓ Minimal demo completed successfully!")
        print(f"Results saved to: demo_results/")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        return False

if __name__ == "__main__":
    print("Cross-Genre Music Style Transfer - Phase 1 Demo")
    print("Testing implementation with current dataset...")
    
    # Test basic functionality
    if test_basic_functionality():
        # Run minimal demo
        run_minimal_demo()
    else:
        print("Basic functionality test failed. Please check your setup.")
        
    print("\nDemo completed!")