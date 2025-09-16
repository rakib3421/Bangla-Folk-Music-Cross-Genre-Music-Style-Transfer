"""
Phase 4.5: Audio Reconstruction Pipeline with Vocal and Rhythm Preservation
==========================================================================

This module implements the complete reconstruction pipeline that combines
processed instrumental tracks with preserved vocals and applies post-processing
for rhythmic coherence and high-quality audio output.

Features:
- Multi-stage audio reconstruction
- Vocal-instrumental recombination with crossfading
- Rhythmic coherence post-processing
- Audio quality enhancement
- Stereo imaging and spatial processing
- Dynamic range optimization
"""

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import Phase 4 components
from source_separation import SourceSeparationPipeline
from vocal_preservation import VocalStyleAdapter
from rhythmic_analysis import RhythmicConstraintSystem

class AudioReconstructionPipeline:
    """
    Complete audio reconstruction pipeline combining all Phase 4 components.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Initialize component pipelines
        self.separator = SourceSeparationPipeline(sample_rate, method='auto')
        self.vocal_adapter = VocalStyleAdapter(sample_rate)
        self.rhythm_system = RhythmicConstraintSystem(sample_rate)
        
        # Reconstruction parameters
        self.crossfade_duration = 0.1  # seconds
        self.enhancement_enabled = True
        
    def separate_and_process(self, audio_path: str, target_genre: str,
                           style_transfer_function: callable,
                           output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: separate, process, and reconstruct audio.
        
        Args:
            audio_path: Path to input audio file
            target_genre: Target genre for style transfer
            style_transfer_function: Function to apply style transfer to instrumental part
            output_dir: Optional output directory
            
        Returns:
            Dictionary with processed audio components
        """
        print(f"Processing {audio_path} for {target_genre} style transfer...")
        
        # Step 1: Source separation
        print("  Step 1: Separating vocals and instruments...")
        separation_result = self.separator.separate_audio(audio_path)
        
        original_audio = separation_result['original']
        vocals = separation_result['vocals']
        instruments = separation_result['instruments']
        
        separation_quality = separation_result['quality_metrics']['overall_quality']
        print(f"    Separation quality: {separation_quality:.3f}")
        
        # Step 2: Vocal style adaptation
        print("  Step 2: Adapting vocal style...")
        vocal_adaptation = self.vocal_adapter.adapt_vocal_style(
            vocals, target_genre, preserve_formants=True
        )
        adapted_vocals = vocal_adaptation['adapted_vocals']
        
        # Step 3: Instrumental style transfer
        print("  Step 3: Applying style transfer to instruments...")
        try:
            # Apply the provided style transfer function
            styled_instruments = style_transfer_function(instruments)
        except Exception as e:
            print(f"    Style transfer failed: {e}, using original instruments")
            styled_instruments = instruments
        
        # Step 4: Rhythmic consistency enforcement
        print("  Step 4: Enforcing rhythmic consistency...")
        rhythmic_analysis = self.rhythm_system.analyze_rhythmic_content(original_audio)
        
        # Apply rhythmic constraints to styled instruments
        rhythmically_consistent_instruments = self.rhythm_system.apply_rhythmic_constraints(
            styled_instruments, target_genre, preserve_original=0.8
        )
        
        # Step 5: Audio reconstruction
        print("  Step 5: Reconstructing final audio...")
        reconstructed_audio = self.reconstruct_audio(
            adapted_vocals, 
            rhythmically_consistent_instruments,
            original_audio
        )
        
        # Step 6: Post-processing and enhancement
        print("  Step 6: Applying post-processing...")
        final_audio = self.post_process_audio(reconstructed_audio, target_genre)
        
        # Prepare results
        results = {
            'original': original_audio,
            'separated_vocals': vocals,
            'separated_instruments': instruments,
            'adapted_vocals': adapted_vocals,
            'styled_instruments': styled_instruments,
            'rhythmic_instruments': rhythmically_consistent_instruments,
            'reconstructed': reconstructed_audio,
            'final': final_audio,
            'separation_quality': separation_quality,
            'rhythmic_analysis': rhythmic_analysis,
            'vocal_adaptation': vocal_adaptation
        }
        
        # Save outputs if directory specified
        if output_dir:
            self.save_reconstruction_results(results, audio_path, target_genre, output_dir)
        
        return results
    
    def reconstruct_audio(self, vocals: np.ndarray, instruments: np.ndarray, 
                         original: np.ndarray) -> np.ndarray:
        """
        Reconstruct audio from vocals and instruments with intelligent mixing.
        
        Args:
            vocals: Processed vocal track
            instruments: Processed instrumental track
            original: Original audio for reference
            
        Returns:
            Reconstructed audio
        """
        # Ensure same length
        min_length = min(len(vocals), len(instruments), len(original))
        vocals = vocals[:min_length]
        instruments = instruments[:min_length]
        original = original[:min_length]
        
        # Analyze original mix balance
        vocal_energy = np.mean(vocals ** 2)
        instrument_energy = np.mean(instruments ** 2)
        total_energy = vocal_energy + instrument_energy
        
        if total_energy > 0:
            vocal_ratio = vocal_energy / total_energy
            instrument_ratio = instrument_energy / total_energy
        else:
            vocal_ratio = 0.3  # Default mix
            instrument_ratio = 0.7
        
        # Apply crossfading at transitions to avoid artifacts
        crossfade_samples = int(self.crossfade_duration * self.sample_rate)
        
        # Simple mixing with energy-based balance
        mixed = vocals * vocal_ratio + instruments * instrument_ratio
        
        # Ensure energy conservation with original
        original_energy = np.mean(original ** 2)
        mixed_energy = np.mean(mixed ** 2)
        
        if mixed_energy > 0:
            energy_correction = np.sqrt(original_energy / mixed_energy)
            mixed = mixed * energy_correction
        
        # Apply soft limiting to prevent clipping
        mixed = np.tanh(mixed * 0.95)
        
        return mixed
    
    def post_process_audio(self, audio: np.ndarray, target_genre: str) -> np.ndarray:
        """
        Apply post-processing for audio enhancement and genre-specific characteristics.
        
        Args:
            audio: Input audio signal
            target_genre: Target genre for processing
            
        Returns:
            Post-processed audio
        """
        if not self.enhancement_enabled:
            return audio
        
        processed = audio.copy()
        
        # Genre-specific EQ curves
        eq_params = {
            'rock': {
                'low_boost': 1.1,      # Slight bass boost
                'mid_boost': 1.2,      # Mid presence
                'high_boost': 1.15     # Brightness
            },
            'jazz': {
                'low_boost': 1.0,      # Natural bass
                'mid_boost': 0.95,     # Slightly scooped mids
                'high_boost': 1.1      # Smooth highs
            },
            'folk': {
                'low_boost': 1.05,     # Warm bass
                'mid_boost': 1.1,      # Present mids
                'high_boost': 1.0      # Natural highs
            }
        }
        
        params = eq_params.get(target_genre.lower(), eq_params['folk'])
        
        # Apply simple EQ using filter banks
        processed = self._apply_eq(processed, params)
        
        # Dynamic range processing
        processed = self._apply_dynamics(processed, target_genre)
        
        # Stereo enhancement (if needed for stereo output)
        # For now, keep mono but apply subtle harmonic enhancement
        processed = self._apply_harmonic_enhancement(processed)
        
        # Final limiting
        processed = self._apply_limiter(processed)
        
        return processed
    
    def _apply_eq(self, audio: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply simple 3-band EQ."""
        # Design filters for low, mid, high bands
        nyquist = self.sample_rate / 2
        
        # Low band: 20-200 Hz
        low_sos = scipy.signal.butter(2, 200/nyquist, btype='low', output='sos')
        low_band = scipy.signal.sosfilt(low_sos, audio)
        
        # Mid band: 200-2000 Hz
        mid_sos = scipy.signal.butter(2, [200/nyquist, 2000/nyquist], btype='band', output='sos')
        mid_band = scipy.signal.sosfilt(mid_sos, audio)
        
        # High band: 2000+ Hz
        high_sos = scipy.signal.butter(2, 2000/nyquist, btype='high', output='sos')
        high_band = scipy.signal.sosfilt(high_sos, audio)
        
        # Apply gains and recombine
        eq_audio = (low_band * params['low_boost'] + 
                   mid_band * params['mid_boost'] + 
                   high_band * params['high_boost'])
        
        return eq_audio
    
    def _apply_dynamics(self, audio: np.ndarray, target_genre: str) -> np.ndarray:
        """Apply dynamic range processing."""
        # Genre-specific dynamic processing
        if target_genre.lower() == 'rock':
            # More compression for rock
            return self._compress_audio(audio, threshold=0.7, ratio=3.0)
        elif target_genre.lower() == 'jazz':
            # Light compression for jazz
            return self._compress_audio(audio, threshold=0.8, ratio=2.0)
        else:
            # Minimal compression for folk
            return self._compress_audio(audio, threshold=0.85, ratio=1.5)
    
    def _compress_audio(self, audio: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
        """Simple audio compressor."""
        # RMS-based compression
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Apply compression to RMS
        compressed_rms = np.where(
            rms > threshold,
            threshold + (rms - threshold) / ratio,
            rms
        )
        
        # Smooth gain changes
        compressed_rms = ndimage.gaussian_filter1d(compressed_rms, sigma=2.0)
        
        # Calculate gain reduction
        gain = compressed_rms / (rms + 1e-10)
        
        # Interpolate gain to audio length
        times = librosa.frames_to_time(np.arange(len(gain)), 
                                     sr=self.sample_rate, hop_length=hop_length)
        audio_times = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        gain_interp = np.interp(audio_times, times, gain)
        
        return audio * gain_interp
    
    def _apply_harmonic_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply subtle harmonic enhancement."""
        # Simple harmonic exciter using soft saturation
        enhanced = audio + 0.1 * np.tanh(audio * 3.0) * 0.5
        
        # High-frequency enhancement
        nyquist = self.sample_rate / 2
        high_sos = scipy.signal.butter(2, 4000/nyquist, btype='high', output='sos')
        high_band = scipy.signal.sosfilt(high_sos, audio)
        
        # Add subtle high-frequency harmonics
        enhanced += 0.05 * np.tanh(high_band * 2.0)
        
        return enhanced
    
    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Apply final limiting to prevent clipping."""
        # Soft limiting
        threshold = 0.95
        limited = np.where(
            np.abs(audio) > threshold,
            threshold * np.tanh(audio / threshold),
            audio
        )
        
        return limited
    
    def save_reconstruction_results(self, results: Dict, audio_path: str, 
                                  target_genre: str, output_dir: str):
        """
        Save all reconstruction results to files.
        
        Args:
            results: Results dictionary
            audio_path: Original audio path
            target_genre: Target genre
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(audio_path).stem
        
        # Save audio files
        audio_files = {
            'original': results['original'],
            'separated_vocals': results['separated_vocals'],
            'separated_instruments': results['separated_instruments'],
            'adapted_vocals': results['adapted_vocals'],
            'styled_instruments': results['styled_instruments'],
            'final_reconstruction': results['final']
        }
        
        for name, audio in audio_files.items():
            filename = f"{base_name}_{target_genre}_{name}.wav"
            sf.write(os.path.join(output_dir, filename), audio, self.sample_rate)
        
        # Save metadata
        metadata = {
            'input_file': audio_path,
            'target_genre': target_genre,
            'separation_quality': results['separation_quality'],
            'rhythmic_analysis': {
                'tempo': results['rhythmic_analysis']['tempo']['tempo'],
                'stability': results['rhythmic_analysis']['rhythmic_stability'],
                'complexity': results['rhythmic_analysis']['complexity_score']
            },
            'vocal_adaptation_params': results['vocal_adaptation']['adaptation_parameters'],
            'processing_chain': [
                'source_separation',
                'vocal_adaptation', 
                'instrumental_style_transfer',
                'rhythmic_consistency',
                'audio_reconstruction',
                'post_processing'
            ]
        }
        
        with open(os.path.join(output_dir, f"{base_name}_{target_genre}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    Results saved to {output_dir}")

class BatchReconstructionProcessor:
    """
    Batch processor for multiple audio files using the reconstruction pipeline.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.pipeline = AudioReconstructionPipeline(sample_rate)
        
    def process_dataset(self, audio_files: List[str], target_genres: List[str],
                       style_transfer_function: callable, output_dir: str) -> Dict:
        """
        Process multiple audio files with different target genres.
        
        Args:
            audio_files: List of audio file paths
            target_genres: List of target genres for each file
            style_transfer_function: Style transfer function
            output_dir: Output directory
            
        Returns:
            Batch processing results
        """
        if len(audio_files) != len(target_genres):
            raise ValueError("Number of audio files must match number of target genres")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        successful = 0
        failed = 0
        
        for i, (audio_file, target_genre) in enumerate(zip(audio_files, target_genres)):
            print(f"\nProcessing {i+1}/{len(audio_files)}: {audio_file} -> {target_genre}")
            
            try:
                file_output_dir = os.path.join(output_dir, f"{Path(audio_file).stem}_{target_genre}")
                
                result = self.pipeline.separate_and_process(
                    audio_file, target_genre, style_transfer_function, file_output_dir
                )
                
                results[f"{audio_file}_{target_genre}"] = {
                    'status': 'success',
                    'separation_quality': result['separation_quality'],
                    'output_dir': file_output_dir
                }
                
                successful += 1
                
            except Exception as e:
                print(f"  Error processing {audio_file}: {e}")
                results[f"{audio_file}_{target_genre}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
                failed += 1
        
        # Save batch summary
        summary = {
            'total_files': len(audio_files),
            'successful': successful,
            'failed': failed,
            'average_separation_quality': np.mean([
                r['separation_quality'] for r in results.values() 
                if r['status'] == 'success'
            ]) if successful > 0 else 0.0,
            'file_results': results
        }
        
        with open(os.path.join(output_dir, 'batch_reconstruction_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBatch processing complete: {successful}/{len(audio_files)} successful")
        print(f"Average separation quality: {summary['average_separation_quality']:.3f}")
        
        return summary

def dummy_style_transfer(audio: np.ndarray) -> np.ndarray:
    """
    Dummy style transfer function for testing.
    In practice, this would be replaced with actual CycleGAN/StarGAN output.
    """
    # Simple processing: apply some filtering and modulation
    # High-pass filter for "brightness"
    from scipy import signal
    sos = signal.butter(2, 500, btype='high', fs=16000, output='sos')
    filtered = signal.sosfilt(sos, audio)
    
    # Mix with original
    return 0.7 * audio + 0.3 * filtered

def test_reconstruction_pipeline():
    """Test the complete reconstruction pipeline."""
    print("Testing Audio Reconstruction Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AudioReconstructionPipeline(sample_rate=16000)
    
    # Find test files
    audio_dir = "data"
    test_files = []
    
    for genre in ['Bangla Folk', 'Jazz', 'Rock']:
        genre_dir = os.path.join(audio_dir, genre)
        if os.path.exists(genre_dir):
            files = [f for f in os.listdir(genre_dir) if f.endswith('.mp3')]
            if files:
                test_files.append(os.path.join(genre_dir, files[0]))
    
    if not test_files:
        print("No audio files found for testing.")
        return
    
    # Test reconstruction pipeline
    output_dir = "experiments/reconstruction_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTesting reconstruction on {len(test_files)} files...")
    
    target_genres = ['rock', 'jazz', 'folk']
    
    for audio_file in test_files[:1]:  # Test with first file only
        for target_genre in target_genres[:2]:  # Test with first 2 genres
            print(f"\nTesting: {audio_file} -> {target_genre}")
            
            try:
                file_output_dir = os.path.join(output_dir, 
                                             f"{Path(audio_file).stem}_{target_genre}")
                
                result = pipeline.separate_and_process(
                    audio_file, target_genre, dummy_style_transfer, file_output_dir
                )
                
                print(f"  Success! Separation quality: {result['separation_quality']:.3f}")
                print(f"  Files saved to: {file_output_dir}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\nReconstruction pipeline test complete. Results in {output_dir}")

if __name__ == "__main__":
    test_reconstruction_pipeline()