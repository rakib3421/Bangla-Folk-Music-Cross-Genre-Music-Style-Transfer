"""
Data Loading and Training Pipeline for Cross-Genre Music Style Transfer
Handles mel-spectrogram preparation, data augmentation, and training orchestration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import random
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from cyclegan_architecture import CycleGAN, StarGAN_VC
from loss_functions import CombinedLoss

class MelSpectrogramDataset(Dataset):
    """
    Dataset for loading and preprocessing mel-spectrograms from audio files.
    """
    
    def __init__(
        self,
        data_dir: str,
        genres: List[str] = ['Bangla Folk', 'Jazz', 'Rock'],
        segment_length: int = 256,  # Time frames
        n_mels: int = 128,
        sr: int = 22050,
        hop_length: int = 512,
        max_files_per_genre: Optional[int] = None,
        augment: bool = True
    ):
        self.data_dir = data_dir
        self.genres = genres
        self.segment_length = segment_length
        self.n_mels = n_mels
        self.sr = sr
        self.hop_length = hop_length
        self.augment = augment
        
        # Create genre to index mapping
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}
        
        # Load file paths
        self.audio_files = self._load_audio_files(max_files_per_genre)
        
        print(f"Loaded {len(self.audio_files)} audio files:")
        for genre in genres:
            count = sum(1 for item in self.audio_files if item['genre'] == genre)
            print(f"  {genre}: {count} files")
    
    def _load_audio_files(self, max_files_per_genre: Optional[int] = None) -> List[Dict]:
        """Load audio file paths organized by genre."""
        audio_files = []
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
        
        for genre in self.genres:
            genre_dir = os.path.join(self.data_dir, genre)
            if not os.path.exists(genre_dir):
                print(f"Warning: Genre directory {genre_dir} not found")
                continue
            
            # Find all audio files in genre directory
            files = []
            for ext in audio_extensions:
                files.extend(Path(genre_dir).glob(f"*{ext}"))
            
            # Limit files per genre if specified
            if max_files_per_genre:
                files = files[:max_files_per_genre]
            
            # Add to dataset
            for file_path in files:
                audio_files.append({
                    'path': str(file_path),
                    'genre': genre,
                    'genre_idx': self.genre_to_idx[genre]
                })
        
        return audio_files
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess audio file.
        
        Returns:
            Dictionary containing mel-spectrogram, genre label, and metadata
        """
        file_info = self.audio_files[idx]
        
        try:
            # Load audio
            y, sr = librosa.load(file_info['path'], sr=self.sr, duration=30.0)  # Load max 30 seconds
            
            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length, n_fft=2048
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [-1, 1] range
            mel_spec_norm = self._normalize_spectrogram(mel_spec_db)
            
            # Extract segment
            mel_segment = self._extract_segment(mel_spec_norm)
            
            # Apply augmentation if enabled
            if self.augment:
                mel_segment = self._apply_augmentation(mel_segment)
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(mel_segment).unsqueeze(0)  # Add channel dimension
            
            return {
                'mel_spectrogram': mel_tensor,
                'genre_label': torch.LongTensor([file_info['genre_idx']]),
                'genre_name': file_info['genre'],
                'file_path': file_info['path']
            }
            
        except Exception as e:
            print(f"Error loading {file_info['path']}: {e}")
            # Return a zero tensor if loading fails
            return {
                'mel_spectrogram': torch.zeros(1, self.n_mels, self.segment_length),
                'genre_label': torch.LongTensor([file_info['genre_idx']]),
                'genre_name': file_info['genre'],
                'file_path': file_info['path']
            }
    
    def _normalize_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Normalize mel-spectrogram to [-1, 1] range."""
        # Normalize to [0, 1] first
        mel_min = mel_spec.min()
        mel_max = mel_spec.max()
        
        if mel_max > mel_min:
            mel_norm = (mel_spec - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_spec)
        
        # Scale to [-1, 1]
        mel_norm = mel_norm * 2.0 - 1.0
        
        return mel_norm
    
    def _extract_segment(self, mel_spec: np.ndarray) -> np.ndarray:
        """Extract a fixed-length segment from mel-spectrogram."""
        n_frames = mel_spec.shape[1]
        
        if n_frames >= self.segment_length:
            # Randomly select a segment
            start_frame = random.randint(0, n_frames - self.segment_length)
            segment = mel_spec[:, start_frame:start_frame + self.segment_length]
        else:
            # Pad if too short
            pad_length = self.segment_length - n_frames
            segment = np.pad(mel_spec, ((0, 0), (0, pad_length)), mode='constant', constant_values=-1.0)
        
        return segment
    
    def _apply_augmentation(self, mel_spec: np.ndarray) -> np.ndarray:
        """Apply data augmentation to mel-spectrogram."""
        if not self.augment:
            return mel_spec
        
        augmented = mel_spec.copy()
        
        # Time masking (random time segments)
        if random.random() < 0.3:
            n_frames = augmented.shape[1]
            mask_length = random.randint(1, min(20, n_frames // 4))
            mask_start = random.randint(0, n_frames - mask_length)
            augmented[:, mask_start:mask_start + mask_length] = -1.0
        
        # Frequency masking (random frequency bands)
        if random.random() < 0.3:
            n_mels = augmented.shape[0]
            mask_length = random.randint(1, min(10, n_mels // 4))
            mask_start = random.randint(0, n_mels - mask_length)
            augmented[mask_start:mask_start + mask_length, :] = -1.0
        
        # Add small amount of noise
        if random.random() < 0.2:
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = np.clip(augmented + noise, -1.0, 1.0)
        
        return augmented

class CrossGenreDataLoader:
    """
    Data loader for cross-genre style transfer.
    Provides paired samples from different domains.
    """
    
    def __init__(
        self,
        dataset: MelSpectrogramDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create separate data loaders for each genre
        self.genre_loaders = {}
        self.genre_iterators = {}
        
        for genre in dataset.genres:
            # Filter dataset by genre
            genre_indices = [i for i, item in enumerate(dataset.audio_files) if item['genre'] == genre]
            genre_subset = torch.utils.data.Subset(dataset, genre_indices)
            
            loader = DataLoader(
                genre_subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=True
            )
            
            self.genre_loaders[genre] = loader
            self.genre_iterators[genre] = iter(loader)
    
    def get_paired_batch(self, genre_A: str, genre_B: str) -> Tuple[Dict, Dict]:
        """
        Get a paired batch from two different genres.
        
        Args:
            genre_A: Source genre name
            genre_B: Target genre name
            
        Returns:
            Tuple of batches from genre A and genre B
        """
        try:
            batch_A = next(self.genre_iterators[genre_A])
        except StopIteration:
            self.genre_iterators[genre_A] = iter(self.genre_loaders[genre_A])
            batch_A = next(self.genre_iterators[genre_A])
        
        try:
            batch_B = next(self.genre_iterators[genre_B])
        except StopIteration:
            self.genre_iterators[genre_B] = iter(self.genre_loaders[genre_B])
            batch_B = next(self.genre_iterators[genre_B])
        
        return batch_A, batch_B

class CycleGANTrainer:
    """
    Training pipeline for CycleGAN music style transfer.
    """
    
    def __init__(
        self,
        model: CycleGAN,
        data_loader: CrossGenreDataLoader,
        device: torch.device = None,
        lr: float = 0.0002,
        betas: Tuple[float, float] = (0.5, 0.999),
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            list(self.model.G_AB.parameters()) + list(self.model.G_BA.parameters()),
            lr=lr, betas=betas
        )
        
        self.optimizer_D_A = optim.Adam(self.model.D_A.parameters(), lr=lr, betas=betas)
        self.optimizer_D_B = optim.Adam(self.model.D_B.parameters(), lr=lr, betas=betas)
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            lambda_cycle=lambda_cycle,
            lambda_identity=lambda_identity
        )
        
        # Training statistics
        self.training_stats = {
            'generator_losses': [],
            'discriminator_losses': [],
            'cycle_losses': [],
            'identity_losses': []
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"CycleGAN Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Save directory: {save_dir}")
        print(f"  Learning rate: {lr}")
    
    def train_epoch(
        self,
        epoch: int,
        genre_A: str = 'Bangla Folk',
        genre_B: str = 'Rock',
        n_batches: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            genre_A: Source genre
            genre_B: Target genre
            n_batches: Number of batches to train on
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'generator_total': 0.0,
            'generator_adversarial': 0.0,
            'generator_cycle': 0.0,
            'generator_identity': 0.0,
            'discriminator_total': 0.0,
            'discriminator_A': 0.0,
            'discriminator_B': 0.0
        }
        
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}")
        
        for batch_idx in pbar:
            try:
                # Get paired batch
                batch_A, batch_B = self.data_loader.get_paired_batch(genre_A, genre_B)
                
                # Move to device
                real_A = batch_A['mel_spectrogram'].to(self.device)
                real_B = batch_B['mel_spectrogram'].to(self.device)
                
                # Skip if batch sizes don't match
                if real_A.size(0) != real_B.size(0):
                    continue
                
                # Train Generators
                self.optimizer_G.zero_grad()
                
                # Generate fake samples
                fake_B = self.model.G_AB(real_A)
                fake_A = self.model.G_BA(real_B)
                
                # Cycle consistency
                cycle_A = self.model.G_BA(fake_B)
                cycle_B = self.model.G_AB(fake_A)
                
                # Identity mapping
                identity_A = self.model.G_BA(real_A)
                identity_B = self.model.G_AB(real_B)
                
                # Discriminator outputs for generator training
                D_A_fake = self.model.D_A(fake_A)
                D_B_fake = self.model.D_B(fake_B)
                
                # Create output dict for generator loss
                gen_output = {
                    'fake_A': fake_A, 'fake_B': fake_B,
                    'cycle_A': cycle_A, 'cycle_B': cycle_B,
                    'identity_A': identity_A, 'identity_B': identity_B,
                    'real_A': real_A, 'real_B': real_B,
                    'D_A_fake': D_A_fake, 'D_B_fake': D_B_fake
                }
                
                generator_losses = self.criterion.compute_generator_loss(gen_output)
                generator_losses['total'].backward()
                self.optimizer_G.step()
                
                # Train Discriminator A
                self.optimizer_D_A.zero_grad()
                
                # Fresh forward pass for discriminator with detached fake samples
                with torch.no_grad():
                    fake_A_detached = self.model.G_BA(real_B)
                
                D_A_real = self.model.D_A(real_A)
                D_A_fake_detached = self.model.D_A(fake_A_detached)
                
                disc_A_output = {
                    'D_A_real': D_A_real,
                    'D_A_fake': D_A_fake_detached,
                    'D_B_real': torch.zeros_like(D_A_real),  # Dummy
                    'D_B_fake': torch.zeros_like(D_A_real)   # Dummy
                }
                
                discriminator_losses_A = self.criterion.compute_discriminator_loss(disc_A_output)
                discriminator_losses_A['D_A'].backward()
                self.optimizer_D_A.step()
                
                # Train Discriminator B
                self.optimizer_D_B.zero_grad()
                
                with torch.no_grad():
                    fake_B_detached = self.model.G_AB(real_A)
                
                D_B_real = self.model.D_B(real_B)
                D_B_fake_detached = self.model.D_B(fake_B_detached)
                
                disc_B_output = {
                    'D_A_real': torch.zeros_like(D_B_real),  # Dummy
                    'D_A_fake': torch.zeros_like(D_B_real),  # Dummy
                    'D_B_real': D_B_real,
                    'D_B_fake': D_B_fake_detached
                }
                
                discriminator_losses_B = self.criterion.compute_discriminator_loss(disc_B_output)
                discriminator_losses_B['D_B'].backward()
                self.optimizer_D_B.step()
                
                # Combine discriminator losses for reporting
                discriminator_losses = {
                    'D_A': discriminator_losses_A['D_A'],
                    'D_B': discriminator_losses_B['D_B'],
                    'total': discriminator_losses_A['D_A'] + discriminator_losses_B['D_B']
                }
                
                # Update statistics
                epoch_losses['generator_total'] += generator_losses['total'].item()
                epoch_losses['generator_adversarial'] += generator_losses['adversarial'].item()
                epoch_losses['generator_cycle'] += generator_losses['cycle'].item()
                epoch_losses['generator_identity'] += generator_losses['identity'].item()
                epoch_losses['discriminator_total'] += discriminator_losses['total'].item()
                epoch_losses['discriminator_A'] += discriminator_losses['D_A'].item()
                epoch_losses['discriminator_B'] += discriminator_losses['D_B'].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'G_loss': f"{generator_losses['total'].item():.4f}",
                    'D_loss': f"{discriminator_losses['total'].item():.4f}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def train(
        self,
        n_epochs: int,
        genre_pairs: List[Tuple[str, str]] = [('Bangla Folk', 'Rock'), ('Bangla Folk', 'Jazz')],
        save_every: int = 10,
        n_batches_per_epoch: int = 100
    ):
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs to train
            genre_pairs: List of genre pairs for training
            save_every: Save model every N epochs
            n_batches_per_epoch: Number of batches per epoch
        """
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Genre pairs: {genre_pairs}")
        print(f"Batches per epoch: {n_batches_per_epoch}")
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            epoch_stats = {}
            
            # Train on each genre pair
            for genre_A, genre_B in genre_pairs:
                print(f"Training {genre_A} <-> {genre_B}")
                
                losses = self.train_epoch(
                    epoch=epoch,
                    genre_A=genre_A,
                    genre_B=genre_B,
                    n_batches=n_batches_per_epoch
                )
                
                epoch_stats[f"{genre_A}_{genre_B}"] = losses
                
                # Print losses
                print(f"  Generator Loss: {losses['generator_total']:.4f}")
                print(f"  Discriminator Loss: {losses['discriminator_total']:.4f}")
                print(f"  Cycle Loss: {losses['generator_cycle']:.4f}")
            
            # Save model checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, epoch_stats)
            
            # Update training statistics
            avg_gen_loss = np.mean([stats['generator_total'] for stats in epoch_stats.values()])
            avg_disc_loss = np.mean([stats['discriminator_total'] for stats in epoch_stats.values()])
            
            self.training_stats['generator_losses'].append(avg_gen_loss)
            self.training_stats['discriminator_losses'].append(avg_disc_loss)
        
        print(f"\nTraining completed!")
        self.save_checkpoint(n_epochs, {}, final=True)
    
    def save_checkpoint(self, epoch: int, epoch_stats: Dict, final: bool = False):
        """Save model checkpoint and training statistics."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'training_stats': self.training_stats,
            'epoch_stats': epoch_stats
        }
        
        filename = f"cyclegan_epoch_{epoch}.pth" if not final else "cyclegan_final.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        # Save training stats as JSON
        stats_file = os.path.join(self.save_dir, f"training_stats_epoch_{epoch}.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'epoch': epoch,
                'training_stats': self.training_stats,
                'epoch_stats': epoch_stats
            }, f, indent=2)

def create_demo_training_setup():
    """Create a demo training setup for testing."""
    print("Creating demo training setup...")
    
    # Create dataset
    dataset = MelSpectrogramDataset(
        data_dir="data",
        genres=['Bangla Folk', 'Jazz', 'Rock'],
        max_files_per_genre=5,  # Small number for demo
        segment_length=128,  # Smaller for faster processing
        n_mels=64,  # Reduced for demo
        augment=True
    )
    
    # Create data loader
    data_loader = CrossGenreDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True
    )
    
    # Create model
    model = CycleGAN(
        input_channels=1,
        base_generator_channels=32,  # Reduced for demo
        base_discriminator_channels=32,
        n_residual_blocks=3,  # Reduced for demo
        input_height=64,
        input_width=128
    )
    
    # Create trainer
    trainer = CycleGANTrainer(
        model=model,
        data_loader=data_loader,
        device=torch.device('cpu'),  # Use CPU for demo
        lr=0.0002,
        save_dir="demo_checkpoints"
    )
    
    return trainer

def test_training_pipeline():
    """Test the training pipeline with minimal setup."""
    print("Testing training pipeline...")
    
    try:
        # Create demo setup
        trainer = create_demo_training_setup()
        
        # Test one epoch
        print("\nTesting single epoch...")
        losses = trainer.train_epoch(
            epoch=1,
            genre_A='Bangla Folk',
            genre_B='Rock',
            n_batches=3  # Very small for testing
        )
        
        print(f"✓ Training test completed!")
        print(f"  Generator loss: {losses['generator_total']:.4f}")
        print(f"  Discriminator loss: {losses['discriminator_total']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False

if __name__ == "__main__":
    test_training_pipeline()