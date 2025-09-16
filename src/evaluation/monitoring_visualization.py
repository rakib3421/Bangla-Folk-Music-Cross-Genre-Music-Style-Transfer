"""
Comprehensive Monitoring and Visualization System for Cross-Genre Music Style Transfer
Implements TensorBoard logging, metrics visualization, and audio sample generation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use(['seaborn-v0_8', 'fast'])
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Union
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class VisualizationTools:
    """
    Comprehensive visualization tools for monitoring training progress.
    """
    
    def __init__(self, experiment_dir: str, writer: SummaryWriter = None):
        """
        Initialize visualization tools.
        
        Args:
            experiment_dir: Directory for experiment outputs
            writer: TensorBoard writer (optional)
        """
        self.experiment_dir = experiment_dir
        self.writer = writer
        
        # Create visualization directories
        os.makedirs(f"{experiment_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{experiment_dir}/audio_samples", exist_ok=True)
        
        print(f"Visualization tools initialized for: {experiment_dir}")
    
    def plot_mel_spectrogram(
        self, 
        mel_spec: torch.Tensor, 
        title: str = "Mel Spectrogram",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram tensor [mels, time]
            title: Plot title
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to numpy and handle batch dimension
        if mel_spec.dim() == 3:
            mel_spec = mel_spec[0]  # Take first batch
        mel_data = mel_spec.detach().cpu().numpy()
        
        # Plot spectrogram
        img = ax.imshow(
            mel_data, 
            aspect='auto', 
            origin='lower',
            interpolation='nearest',
            cmap='viridis'
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Mel Bins', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved mel spectrogram plot: {save_path}")
        
        return fig
    
    def plot_training_curves(
        self, 
        history: Dict[str, List[float]], 
        title: str = "Training Curves",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot training loss curves.
        
        Args:
            history: Dictionary of training metrics
            title: Plot title
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Generator and Discriminator losses
        if 'generator_loss' in history and 'discriminator_A_loss' in history:
            ax = axes[0, 0]
            epochs = range(1, len(history['generator_loss']) + 1)
            
            ax.plot(epochs, history['generator_loss'], 'b-', label='Generator', linewidth=2)
            ax.plot(epochs, history['discriminator_A_loss'], 'r-', label='Discriminator A', linewidth=2)
            if 'discriminator_B_loss' in history:
                ax.plot(epochs, history['discriminator_B_loss'], 'g-', label='Discriminator B', linewidth=2)
            
            ax.set_title('Adversarial Losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Cycle consistency loss
        if 'cycle_loss' in history:
            ax = axes[0, 1]
            epochs = range(1, len(history['cycle_loss']) + 1)
            
            ax.plot(epochs, history['cycle_loss'], 'purple', linewidth=2)
            ax.set_title('Cycle Consistency Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        # Identity loss
        if 'identity_loss' in history:
            ax = axes[1, 0]
            epochs = range(1, len(history['identity_loss']) + 1)
            
            ax.plot(epochs, history['identity_loss'], 'orange', linewidth=2)
            ax.set_title('Identity Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        # Combined losses
        if len(history) > 0:
            ax = axes[1, 1]
            
            for key, values in history.items():
                if 'loss' in key.lower():
                    epochs = range(1, len(values) + 1)
                    ax.plot(epochs, values, label=key.replace('_', ' ').title(), alpha=0.7)
            
            ax.set_title('All Losses (Log Scale)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (log)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves: {save_path}")
        
        return fig
    
    def plot_style_transfer_comparison(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        genre_A: str = "Genre A",
        genre_B: str = "Genre B",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot style transfer results comparison.
        
        Args:
            real_A: Real samples from domain A
            fake_B: Generated samples A→B
            real_B: Real samples from domain B
            fake_A: Generated samples B→A
            genre_A: Name of genre A
            genre_B: Name of genre B
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Convert tensors to numpy
        specs = [real_A, fake_B, real_B, fake_A]
        titles = [f"Real {genre_A}", f"Fake {genre_B} (from {genre_A})", 
                 f"Real {genre_B}", f"Fake {genre_A} (from {genre_B})"]
        
        for i, (spec, title) in enumerate(zip(specs, titles)):
            ax = axes[i // 2, i % 2]
            
            # Handle batch dimension
            if spec.dim() == 4:
                spec = spec[0, 0]  # Take first sample, first channel
            elif spec.dim() == 3:
                spec = spec[0]
            
            spec_data = spec.detach().cpu().numpy()
            
            img = ax.imshow(
                spec_data, 
                aspect='auto', 
                origin='lower',
                interpolation='nearest',
                cmap='viridis'
            )
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Bins')
            
            # Add colorbar
            plt.colorbar(img, ax=ax)
        
        plt.suptitle(f'Style Transfer: {genre_A} ↔ {genre_B}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved style transfer comparison: {save_path}")
        
        return fig
    
    def log_to_tensorboard(
        self,
        losses: Dict[str, float],
        step: int,
        phase: str = "train"
    ):
        """
        Log metrics to TensorBoard.
        
        Args:
            losses: Dictionary of loss values
            step: Global step
            phase: Training phase name
        """
        if self.writer is None:
            return
        
        for name, value in losses.items():
            self.writer.add_scalar(f'{phase}/{name}', value, step)
    
    def log_spectrograms_to_tensorboard(
        self,
        spectrograms: Dict[str, torch.Tensor],
        step: int,
        phase: str = "train"
    ):
        """
        Log spectrograms to TensorBoard.
        
        Args:
            spectrograms: Dictionary of spectrogram tensors
            step: Global step
            phase: Training phase name
        """
        if self.writer is None:
            return
        
        for name, spec in spectrograms.items():
            # Convert to image format [1, H, W]
            if spec.dim() == 4:
                spec = spec[0, 0]  # Take first sample, first channel
            elif spec.dim() == 3:
                spec = spec[0]
            
            # Normalize for visualization
            spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            spec_img = spec_norm.unsqueeze(0)  # Add channel dimension
            
            self.writer.add_image(f'{phase}/{name}', spec_img, step)

class AudioSampleGenerator:
    """
    Generate audio samples from mel spectrograms for evaluation.
    """
    
    def __init__(
        self, 
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Initialize audio sample generator.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel bins
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Initialize mel basis for inverse transform
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=80,
            fmax=sample_rate // 2
        )
        
        print(f"AudioSampleGenerator initialized:")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Mel bins: {n_mels}")
    
    def mel_to_audio(self, mel_spec: torch.Tensor) -> np.ndarray:
        """
        Convert mel spectrogram back to audio using Griffin-Lim algorithm.
        
        Args:
            mel_spec: Mel spectrogram tensor [mels, time]
            
        Returns:
            Audio waveform as numpy array
        """
        # Convert to numpy
        if isinstance(mel_spec, torch.Tensor):
            mel_spec = mel_spec.detach().cpu().numpy()
        
        # Handle batch dimension
        if mel_spec.ndim == 3:
            mel_spec = mel_spec[0]
        
        # Convert from dB to linear scale
        mel_linear = librosa.db_to_power(mel_spec)
        
        # Inverse mel transform to get magnitude spectrogram
        mag_spec = np.dot(self.mel_basis.T, mel_linear)
        
        # Apply Griffin-Lim algorithm to get audio
        audio = librosa.griffinlim(
            mag_spec,
            n_iter=32,
            hop_length=self.hop_length,
            win_length=self.n_fft
        )
        
        return audio
    
    def save_audio_sample(
        self, 
        mel_spec: torch.Tensor, 
        filename: str, 
        output_dir: str
    ):
        """
        Save mel spectrogram as audio file.
        
        Args:
            mel_spec: Mel spectrogram tensor
            filename: Output filename
            output_dir: Output directory
        """
        # Convert to audio
        audio = self.mel_to_audio(mel_spec)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save to file
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, audio, self.sample_rate)
        
        print(f"Saved audio sample: {output_path}")
        return output_path
    
    def generate_style_transfer_samples(
        self,
        model,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        output_dir: str,
        epoch: int,
        genre_A: str = "A",
        genre_B: str = "B"
    ):
        """
        Generate and save style transfer audio samples.
        
        Args:
            model: Trained model
            real_A: Real samples from domain A
            real_B: Real samples from domain B
            output_dir: Output directory
            epoch: Current epoch number
            genre_A: Name of genre A
            genre_B: Name of genre B
        """
        model.eval()
        
        with torch.no_grad():
            # Generate fake samples
            fake_B = model.G_AB(real_A)
            fake_A = model.G_BA(real_B)
            
            # Create epoch directory
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Save samples (take first sample from batch)
            self.save_audio_sample(
                real_A[0], 
                f"real_{genre_A.lower().replace(' ', '_')}.wav", 
                epoch_dir
            )
            
            self.save_audio_sample(
                fake_B[0], 
                f"fake_{genre_B.lower().replace(' ', '_')}_from_{genre_A.lower().replace(' ', '_')}.wav", 
                epoch_dir
            )
            
            self.save_audio_sample(
                real_B[0], 
                f"real_{genre_B.lower().replace(' ', '_')}.wav", 
                epoch_dir
            )
            
            self.save_audio_sample(
                fake_A[0], 
                f"fake_{genre_A.lower().replace(' ', '_')}_from_{genre_B.lower().replace(' ', '_')}.wav", 
                epoch_dir
            )
        
        model.train()

class TrainingMonitor:
    """
    Comprehensive training monitoring with visualization and audio generation.
    """
    
    def __init__(
        self, 
        experiment_dir: str,
        sample_rate: int = 22050,
        n_mels: int = 128
    ):
        """
        Initialize training monitor.
        
        Args:
            experiment_dir: Experiment directory
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
        """
        self.experiment_dir = experiment_dir
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(f"{experiment_dir}/logs")
        
        # Initialize visualization tools
        self.viz_tools = VisualizationTools(experiment_dir, self.writer)
        
        # Initialize audio generator
        self.audio_generator = AudioSampleGenerator(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        
        # Training statistics
        self.training_stats = {
            'total_epochs': 0,
            'total_batches': 0,
            'best_generator_loss': float('inf'),
            'best_discriminator_loss': float('inf'),
            'training_start_time': time.time()
        }
        
        print(f"TrainingMonitor initialized for: {experiment_dir}")
    
    def log_epoch_results(
        self,
        epoch: int,
        losses: Dict[str, float],
        model = None,
        sample_batch: Dict[str, torch.Tensor] = None,
        phase: str = "train",
        genre_A: str = "A",
        genre_B: str = "B"
    ):
        """
        Log comprehensive epoch results.
        
        Args:
            epoch: Current epoch
            losses: Dictionary of loss values
            model: Trained model (for sample generation)
            sample_batch: Sample batch for visualization
            phase: Training phase
            genre_A: Name of genre A
            genre_B: Name of genre B
        """
        # Log scalar metrics
        self.viz_tools.log_to_tensorboard(losses, epoch, phase)
        
        # Update statistics
        self.training_stats['total_epochs'] = epoch
        if 'generator' in losses:
            if losses['generator'] < self.training_stats['best_generator_loss']:
                self.training_stats['best_generator_loss'] = losses['generator']
        
        # Generate visualizations periodically
        if epoch % 10 == 0 and sample_batch is not None:
            # Generate style transfer samples
            if model is not None and hasattr(model, 'G_AB'):
                try:
                    self.audio_generator.generate_style_transfer_samples(
                        model=model,
                        real_A=sample_batch['real_A'],
                        real_B=sample_batch['real_B'],
                        output_dir=f"{self.experiment_dir}/audio_samples",
                        epoch=epoch,
                        genre_A=genre_A,
                        genre_B=genre_B
                    )
                except Exception as e:
                    print(f"Audio sample generation failed: {e}")
            
            # Generate visualizations
            try:
                if model is not None and hasattr(model, 'G_AB'):
                    with torch.no_grad():
                        fake_B = model.G_AB(sample_batch['real_A'])
                        fake_A = model.G_BA(sample_batch['real_B'])
                    
                    # Create comparison plot
                    fig = self.viz_tools.plot_style_transfer_comparison(
                        real_A=sample_batch['real_A'],
                        fake_B=fake_B,
                        real_B=sample_batch['real_B'],
                        fake_A=fake_A,
                        genre_A=genre_A,
                        genre_B=genre_B,
                        save_path=f"{self.experiment_dir}/visualizations/{phase}_epoch_{epoch:03d}_comparison.png"
                    )
                    plt.close(fig)
                    
                    # Log spectrograms to TensorBoard
                    spectrograms = {
                        f'real_{genre_A}': sample_batch['real_A'],
                        f'fake_{genre_B}': fake_B,
                        f'real_{genre_B}': sample_batch['real_B'],
                        f'fake_{genre_A}': fake_A
                    }
                    self.viz_tools.log_spectrograms_to_tensorboard(spectrograms, epoch, phase)
                
            except Exception as e:
                print(f"Visualization generation failed: {e}")
    
    def save_training_summary(self, training_history: Dict[str, List[float]], phase: str):
        """
        Save comprehensive training summary.
        
        Args:
            training_history: Complete training history
            phase: Training phase name
        """
        # Save training curves
        fig = self.viz_tools.plot_training_curves(
            history=training_history,
            title=f"Training Curves - {phase}",
            save_path=f"{self.experiment_dir}/visualizations/{phase}_training_curves.png"
        )
        plt.close(fig)
        
        # Update and save statistics
        self.training_stats['training_end_time'] = time.time()
        self.training_stats['total_training_time'] = (
            self.training_stats['training_end_time'] - 
            self.training_stats['training_start_time']
        )
        
        stats_path = f"{self.experiment_dir}/training_statistics_{phase}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Training summary saved: {stats_path}")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

def test_monitoring_system():
    """Test the monitoring and visualization system."""
    print("Testing Monitoring and Visualization System...")
    
    try:
        # Create test experiment directory
        experiment_dir = "test_monitoring_experiment"
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize monitor
        monitor = TrainingMonitor(experiment_dir)
        
        # Generate dummy data
        batch_size = 4
        n_mels = 128
        time_steps = 128
        
        dummy_real_A = torch.randn(batch_size, 1, n_mels, time_steps)
        dummy_real_B = torch.randn(batch_size, 1, n_mels, time_steps)
        
        # Test visualization tools
        print("  Testing mel spectrogram plotting...")
        fig = monitor.viz_tools.plot_mel_spectrogram(
            dummy_real_A[0, 0],
            title="Test Mel Spectrogram",
            save_path=f"{experiment_dir}/test_mel_spec.png"
        )
        plt.close(fig)
        
        # Test training curves
        print("  Testing training curve plotting...")
        dummy_history = {
            'generator_loss': np.random.exponential(1, 50).tolist(),
            'discriminator_A_loss': np.random.exponential(0.5, 50).tolist(),
            'discriminator_B_loss': np.random.exponential(0.5, 50).tolist(),
            'cycle_loss': np.random.exponential(2, 50).tolist(),
            'identity_loss': np.random.exponential(0.1, 50).tolist()
        }
        
        fig = monitor.viz_tools.plot_training_curves(
            dummy_history,
            title="Test Training Curves",
            save_path=f"{experiment_dir}/test_training_curves.png"
        )
        plt.close(fig)
        
        # Test style transfer comparison
        print("  Testing style transfer comparison...")
        fig = monitor.viz_tools.plot_style_transfer_comparison(
            real_A=dummy_real_A,
            fake_B=dummy_real_B,  # Using dummy data
            real_B=dummy_real_B,
            fake_A=dummy_real_A,  # Using dummy data
            genre_A="Folk",
            genre_B="Rock",
            save_path=f"{experiment_dir}/test_style_transfer.png"
        )
        plt.close(fig)
        
        # Test audio generation
        print("  Testing audio sample generation...")
        audio_gen = AudioSampleGenerator()
        audio = audio_gen.mel_to_audio(dummy_real_A[0, 0])
        print(f"    Generated audio shape: {audio.shape}")
        
        # Save test audio
        output_path = audio_gen.save_audio_sample(
            dummy_real_A[0, 0],
            "test_audio.wav",
            experiment_dir
        )
        
        # Test monitoring logging
        print("  Testing monitoring logs...")
        dummy_losses = {
            'generator': 1.5,
            'discriminator_A': 0.8,
            'discriminator_B': 0.7,
            'cycle': 2.1,
            'identity': 0.3
        }
        
        monitor.log_epoch_results(
            epoch=1,
            losses=dummy_losses,
            phase="test"
        )
        
        # Close monitor
        monitor.close()
        
        print(f"\n✓ Monitoring system test completed successfully!")
        print(f"  Test outputs saved to: {experiment_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Monitoring system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_monitoring_system()