"""
CPU-Optimized Training Script for Cross-Genre Music Style Transfer
Complete training pipeline optimized for CPU-only systems.
"""

import os
import sys
import time
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from cpu_optimization import CPUOptimizedConfig, cpu_optimized_training_step
from cyclegan_architecture import CycleGAN
from loss_functions import CombinedLoss
from training_pipeline import MelSpectrogramDataset, CrossGenreDataLoader
from monitoring_visualization import TrainingMonitor

class CPUOptimizedTrainer:
    """
    CPU-optimized trainer for cross-genre music style transfer.
    """
    
    def __init__(
        self, 
        data_dir: str = "data",
        experiment_name: str = None
    ):
        """
        Initialize CPU-optimized trainer.
        
        Args:
            data_dir: Directory containing audio data
            experiment_name: Name for this experiment
        """
        self.data_dir = data_dir
        self.experiment_name = experiment_name or f"cpu_experiment_{int(time.time())}"
        
        # Initialize CPU configuration
        self.config = CPUOptimizedConfig()
        self.config.optimize_torch_for_cpu()
        
        # Create experiment directory
        self.experiment_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        
        # Initialize components
        self.setup_data()
        self.setup_models()
        self.setup_optimizers()
        self.setup_criterion()
        self.setup_monitor()
        
        print(f"CPU-Optimized Trainer initialized:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Device: {self.config.device}")
        print(f"  Dataset size: {len(self.dataset)} samples")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self):
        """Setup dataset and data loader."""
        print("Setting up CPU-optimized dataset...")
        
        dataset_kwargs = self.config.get_dataset_kwargs()
        
        self.dataset = MelSpectrogramDataset(
            data_dir=self.data_dir,
            genres=['Bangla Folk', 'Jazz', 'Rock'],
            **dataset_kwargs
        )
        
        self.data_loader = CrossGenreDataLoader(
            dataset=self.dataset,
            batch_size=self.config.training_config['batch_size'],
            shuffle=True,
            num_workers=0  # CPU optimization
        )
    
    def setup_models(self):
        """Setup CycleGAN model."""
        print("Setting up CPU-optimized CycleGAN model...")
        
        self.model = CycleGAN(
            input_channels=1,
            base_generator_channels=self.config.model_config['base_generator_channels'],
            base_discriminator_channels=self.config.model_config['base_discriminator_channels'],
            n_residual_blocks=self.config.model_config['n_residual_blocks'],
            input_height=self.config.model_config['input_height'],
            input_width=self.config.model_config['input_width']
        ).to(self.config.device)
    
    def setup_optimizers(self):
        """Setup optimizers."""
        print("Setting up optimizers...")
        
        lr = self.config.training_config['learning_rate']
        
        self.optimizer_G = optim.Adam(
            list(self.model.G_AB.parameters()) + list(self.model.G_BA.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        
        self.optimizer_D_A = optim.Adam(
            self.model.D_A.parameters(),
            lr=lr, betas=(0.5, 0.999)
        )
        
        self.optimizer_D_B = optim.Adam(
            self.model.D_B.parameters(),
            lr=lr, betas=(0.5, 0.999)
        )
        
        self.optimizers = {
            'G': self.optimizer_G,
            'D_A': self.optimizer_D_A,
            'D_B': self.optimizer_D_B
        }
        
        # Learning rate schedulers
        self.schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.5
            )
    
    def setup_criterion(self):
        """Setup loss criterion."""
        print("Setting up loss criterion...")
        
        loss_kwargs = self.config.get_loss_kwargs()
        
        self.criterion = CombinedLoss(
            lambda_cycle=loss_kwargs['lambda_cycle'],
            lambda_identity=loss_kwargs['lambda_identity'],
            lambda_perceptual=loss_kwargs['lambda_perceptual'],
            lambda_rhythm=loss_kwargs['lambda_rhythm'],
            lambda_spectral=loss_kwargs['lambda_spectral']
        )
    
    def setup_monitor(self):
        """Setup training monitor."""
        print("Setting up training monitor...")
        
        self.monitor = TrainingMonitor(
            experiment_dir=self.experiment_dir,
            sample_rate=self.config.audio_config['sample_rate'],
            n_mels=self.config.model_config['n_mels']
        )
    
    def train_epoch(
        self, 
        epoch: int, 
        genre_A: str = 'Bangla Folk', 
        genre_B: str = 'Rock',
        max_batches: int = 20  # Limit for CPU training
    ) -> Dict[str, float]:
        """
        Train for one epoch with CPU optimizations.
        
        Args:
            epoch: Current epoch number
            genre_A: Source genre
            genre_B: Target genre
            max_batches: Maximum batches per epoch
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        epoch_losses = {
            'generator': 0.0,
            'discriminator_A': 0.0,
            'discriminator_B': 0.0,
            'total_batches': 0
        }
        
        accumulation_steps = self.config.training_config['gradient_accumulation_steps']
        
        for batch_idx in range(max_batches):
            try:
                # Get batch
                batch_A, batch_B = self.data_loader.get_paired_batch(genre_A, genre_B)
                
                # Skip if batch sizes don't match
                if batch_A['mel_spectrogram'].size(0) != batch_B['mel_spectrogram'].size(0):
                    continue
                
                # CPU-optimized training step
                losses = cpu_optimized_training_step(
                    model=self.model,
                    batch_A=batch_A,
                    batch_B=batch_B,
                    optimizers=self.optimizers,
                    criterion=self.criterion,
                    gradient_accumulation_steps=accumulation_steps
                )
                
                if losses is not None:
                    # Update losses
                    for key in ['generator', 'discriminator_A', 'discriminator_B']:
                        epoch_losses[key] += losses[key]
                    epoch_losses['total_batches'] += 1
                
                # Apply gradients every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    for optimizer in self.optimizers.values():
                        optimizer.step()
                        optimizer.zero_grad()
                
            except Exception as e:
                print(f"  Batch {batch_idx} failed: {e}")
                continue
        
        # Final gradient step
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        
        # Average losses
        if epoch_losses['total_batches'] > 0:
            for key in ['generator', 'discriminator_A', 'discriminator_B']:
                epoch_losses[key] /= epoch_losses['total_batches']
        
        return epoch_losses
    
    def train_phase(
        self, 
        phase_name: str,
        genre_A: str, 
        genre_B: str, 
        n_epochs: int,
        max_batches_per_epoch: int = 20
    ) -> Dict[str, List[float]]:
        """
        Train a complete phase.
        
        Args:
            phase_name: Name of training phase
            genre_A: Source genre
            genre_B: Target genre
            n_epochs: Number of epochs
            max_batches_per_epoch: Max batches per epoch
            
        Returns:
            Training history
        """
        print(f"\nTraining {phase_name}: {genre_A} ↔ {genre_B}")
        print(f"Epochs: {n_epochs}, Max batches per epoch: {max_batches_per_epoch}")
        
        history = {
            'generator_loss': [],
            'discriminator_A_loss': [],
            'discriminator_B_loss': []
        }
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            losses = self.train_epoch(
                epoch=epoch + 1,
                genre_A=genre_A,
                genre_B=genre_B,
                max_batches=max_batches_per_epoch
            )
            
            # Update history
            history['generator_loss'].append(losses['generator'])
            history['discriminator_A_loss'].append(losses['discriminator_A'])
            history['discriminator_B_loss'].append(losses['discriminator_B'])
            
            # Update learning rates
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Monitor logging
            try:
                # Get sample batch for monitoring
                sample_batch_A, sample_batch_B = self.data_loader.get_paired_batch(genre_A, genre_B)
                sample_batch = {
                    'real_A': sample_batch_A['mel_spectrogram'],
                    'real_B': sample_batch_B['mel_spectrogram']
                }
                
                self.monitor.log_epoch_results(
                    epoch=epoch + 1,
                    losses=losses,
                    model=self.model,
                    sample_batch=sample_batch,
                    phase=phase_name,
                    genre_A=genre_A,
                    genre_B=genre_B
                )
            except Exception as e:
                print(f"  Monitoring failed: {e}")
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s) - "
                  f"G: {losses['generator']:.4f}, "
                  f"D_A: {losses['discriminator_A']:.4f}, "
                  f"D_B: {losses['discriminator_B']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                checkpoint_path = f"{self.experiment_dir}/checkpoints/{phase_name}_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizers_state_dict': {k: v.state_dict() for k, v in self.optimizers.items()},
                    'history': history,
                    'config': self.config.__dict__
                }, checkpoint_path)
                print(f"    Saved checkpoint: {checkpoint_path}")
        
        return history
    
    def run_training(self):
        """Run complete CPU-optimized training."""
        print(f"\n{'='*70}")
        print(f"STARTING CPU-OPTIMIZED TRAINING: {self.experiment_name}")
        print(f"{'='*70}")
        
        # Save configuration
        self.config.save_config(f"{self.experiment_dir}/config.json")
        
        results = {}
        
        # Phase 1: Folk ↔ Rock
        phase1_epochs = min(self.config.training_config['max_epochs_phase1'], 10)  # Limit for demo
        results['folk_rock'] = self.train_phase(
            phase_name='folk_rock',
            genre_A='Bangla Folk',
            genre_B='Rock',
            n_epochs=phase1_epochs,
            max_batches_per_epoch=15
        )
        
        # Phase 2: Folk ↔ Jazz
        phase2_epochs = min(self.config.training_config['max_epochs_phase1'], 10)  # Limit for demo
        results['folk_jazz'] = self.train_phase(
            phase_name='folk_jazz',
            genre_A='Bangla Folk',
            genre_B='Jazz',
            n_epochs=phase2_epochs,
            max_batches_per_epoch=15
        )
        
        # Save final results
        self.monitor.save_training_summary(results['folk_rock'], 'folk_rock')
        self.monitor.save_training_summary(results['folk_jazz'], 'folk_jazz')
        
        # Close monitor
        self.monitor.close()
        
        print(f"\n✓ CPU-optimized training completed!")
        print(f"Results saved in: {self.experiment_dir}")
        
        return results

def run_cpu_optimized_training():
    """Run CPU-optimized training."""
    print("Starting CPU-Optimized Cross-Genre Music Style Transfer Training...")
    
    try:
        # Create trainer
        trainer = CPUOptimizedTrainer(
            data_dir="data",
            experiment_name="cpu_optimized_training"
        )
        
        # Run training
        results = trainer.run_training()
        
        print(f"\n✓ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_cpu_optimized_training()