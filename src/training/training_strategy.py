"""
Comprehensive Training Strategy for Cross-Genre Music Style Transfer
Implements phased training approach with learning rate scheduling and monitoring.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from cyclegan_architecture import CycleGAN, StarGAN_VC
from loss_functions import CombinedLoss
from training_pipeline import MelSpectrogramDataset, CrossGenreDataLoader
from advanced_preprocessing import AdvancedAudioPreprocessor

class TrainingStrategy:
    """
    Comprehensive training strategy with phased approach and monitoring.
    """
    
    def __init__(
        self,
        data_dir: str,
        experiment_name: str = None,
        base_lr: float = 0.0002,
        batch_size: int = 16,
        device: str = 'auto'
    ):
        """
        Initialize training strategy.
        
        Args:
            data_dir: Directory containing audio data
            experiment_name: Name for this experiment
            base_lr: Base learning rate
            batch_size: Training batch size
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.data_dir = data_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_lr = base_lr
        self.batch_size = batch_size
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create experiment directory
        self.experiment_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/samples", exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f"{self.experiment_dir}/logs")
        
        # Training phases configuration
        self.phases = {
            'phase1_folk_rock': {
                'genres': ('Bangla Folk', 'Rock'),
                'epochs': 50,
                'description': 'Folk ↔ Rock style transfer'
            },
            'phase1_folk_jazz': {
                'genres': ('Bangla Folk', 'Jazz'),
                'epochs': 50,
                'description': 'Folk ↔ Jazz style transfer'
            },
            'phase2_joint': {
                'genres': ('Bangla Folk', 'Rock', 'Jazz'),
                'epochs': 100,
                'description': 'Joint multi-domain training'
            }
        }
        
        print(f"Training Strategy initialized:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Base LR: {base_lr}")
        print(f"  Batch size: {batch_size}")
        print(f"  Experiment dir: {self.experiment_dir}")
    
    def setup_datasets(self, max_files_per_genre: int = None) -> Dict[str, CrossGenreDataLoader]:
        """
        Setup datasets for different training phases.
        
        Args:
            max_files_per_genre: Maximum files per genre for testing
            
        Returns:
            Dictionary of data loaders for each phase
        """
        print("\nSetting up datasets...")
        
        data_loaders = {}
        
        # Phase 1: Individual pairs
        for phase_name in ['phase1_folk_rock', 'phase1_folk_jazz']:
            genres = list(self.phases[phase_name]['genres'])
            
            print(f"  Setting up {phase_name}: {genres}")
            
            dataset = MelSpectrogramDataset(
                data_dir=self.data_dir,
                genres=genres,
                max_files_per_genre=max_files_per_genre,
                segment_length=128,
                n_mels=128,
                augment=True
            )
            
            data_loader = CrossGenreDataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            data_loaders[phase_name] = data_loader
        
        # Phase 2: Joint training
        all_genres = ['Bangla Folk', 'Rock', 'Jazz']
        print(f"  Setting up phase2_joint: {all_genres}")
        
        dataset = MelSpectrogramDataset(
            data_dir=self.data_dir,
            genres=all_genres,
            max_files_per_genre=max_files_per_genre,
            segment_length=128,
            n_mels=128,
            augment=True
        )
        
        data_loader = CrossGenreDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        data_loaders['phase2_joint'] = data_loader
        
        return data_loaders
    
    def setup_models(self) -> Dict[str, nn.Module]:
        """
        Setup models for different training phases.
        
        Returns:
            Dictionary of models
        """
        print("\nSetting up models...")
        
        models = {}
        
        # Phase 1 models: CycleGAN for pairs
        for phase_name in ['phase1_folk_rock', 'phase1_folk_jazz']:
            print(f"  Creating CycleGAN for {phase_name}")
            
            model = CycleGAN(
                input_channels=1,
                base_generator_channels=64,
                base_discriminator_channels=64,
                n_residual_blocks=6,
                input_height=128,
                input_width=128
            ).to(self.device)
            
            models[phase_name] = model
        
        # Phase 2 model: StarGAN-VC for multi-domain
        print(f"  Creating StarGAN-VC for phase2_joint")
        
        model = StarGAN_VC(
            input_channels=1,
            n_domains=3,  # Folk, Rock, Jazz
            base_channels=64,
            n_residual_blocks=6
        ).to(self.device)
        
        models['phase2_joint'] = model
        
        return models
    
    def setup_optimizers(self, models: Dict[str, nn.Module]) -> Dict[str, Dict[str, optim.Optimizer]]:
        """
        Setup optimizers for all models.
        
        Args:
            models: Dictionary of models
            
        Returns:
            Dictionary of optimizers
        """
        print("\nSetting up optimizers...")
        
        optimizers = {}
        
        for phase_name, model in models.items():
            print(f"  Setting up optimizers for {phase_name}")
            
            if 'phase1' in phase_name:
                # CycleGAN optimizers
                optimizer_G = optim.Adam(
                    list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
                    lr=self.base_lr,
                    betas=(0.5, 0.999)
                )
                
                optimizer_D_A = optim.Adam(
                    model.D_A.parameters(),
                    lr=self.base_lr,
                    betas=(0.5, 0.999)
                )
                
                optimizer_D_B = optim.Adam(
                    model.D_B.parameters(),
                    lr=self.base_lr,
                    betas=(0.5, 0.999)
                )
                
                optimizers[phase_name] = {
                    'G': optimizer_G,
                    'D_A': optimizer_D_A,
                    'D_B': optimizer_D_B
                }
            
            else:
                # StarGAN-VC optimizers
                optimizer_G = optim.Adam(
                    model.generator.parameters(),
                    lr=self.base_lr,
                    betas=(0.5, 0.999)
                )
                
                optimizer_D = optim.Adam(
                    model.discriminator.parameters(),
                    lr=self.base_lr,
                    betas=(0.5, 0.999)
                )
                
                optimizers[phase_name] = {
                    'G': optimizer_G,
                    'D': optimizer_D
                }
        
        return optimizers
    
    def setup_schedulers(self, optimizers: Dict[str, Dict[str, optim.Optimizer]]) -> Dict[str, Dict[str, optim.lr_scheduler.LRScheduler]]:
        """
        Setup learning rate schedulers.
        
        Args:
            optimizers: Dictionary of optimizers
            
        Returns:
            Dictionary of schedulers
        """
        print("\nSetting up learning rate schedulers...")
        
        schedulers = {}
        
        for phase_name, phase_optimizers in optimizers.items():
            print(f"  Setting up schedulers for {phase_name}")
            
            phase_schedulers = {}
            
            for opt_name, optimizer in phase_optimizers.items():
                # Exponential decay: 0.5 every 100 epochs
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=100,
                    gamma=0.5
                )
                phase_schedulers[opt_name] = scheduler
            
            schedulers[phase_name] = phase_schedulers
        
        return schedulers
    
    def setup_criteria(self) -> Dict[str, CombinedLoss]:
        """
        Setup loss criteria for different phases.
        
        Returns:
            Dictionary of loss criteria
        """
        print("\nSetting up loss criteria...")
        
        criteria = {}
        
        for phase_name in self.phases.keys():
            print(f"  Setting up criterion for {phase_name}")
            
            criterion = CombinedLoss(
                lambda_cycle=10.0,
                lambda_identity=0.5,
                lambda_perceptual=1.0,
                lambda_rhythm=1.0,
                lambda_spectral=1.0,
                adversarial_loss_type='lsgan'
            )
            
            criteria[phase_name] = criterion
        
        return criteria
    
    def train_phase1_cyclegan(
        self,
        phase_name: str,
        model: CycleGAN,
        data_loader: CrossGenreDataLoader,
        optimizers: Dict[str, optim.Optimizer],
        schedulers: Dict[str, optim.lr_scheduler.LRScheduler],
        criterion: CombinedLoss,
        n_epochs: int
    ) -> Dict[str, List[float]]:
        """
        Train CycleGAN for Phase 1.
        
        Args:
            phase_name: Name of the training phase
            model: CycleGAN model
            data_loader: Data loader
            optimizers: Dictionary of optimizers
            schedulers: Dictionary of schedulers
            criterion: Loss criterion
            n_epochs: Number of epochs
            
        Returns:
            Training history
        """
        print(f"\nTraining {phase_name} for {n_epochs} epochs...")
        
        model.train()
        history = {
            'generator_loss': [],
            'discriminator_A_loss': [],
            'discriminator_B_loss': [],
            'cycle_loss': [],
            'identity_loss': []
        }
        
        genre_A, genre_B = self.phases[phase_name]['genres']
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            epoch_losses = {
                'generator': 0.0,
                'discriminator_A': 0.0,
                'discriminator_B': 0.0,
                'cycle': 0.0,
                'identity': 0.0
            }
            
            n_batches = min(50, len(data_loader.dataset) // self.batch_size)  # Limit for demo
            
            for batch_idx in range(n_batches):
                try:
                    # Get batch
                    batch_A, batch_B = data_loader.get_paired_batch(genre_A, genre_B)
                    real_A = batch_A['mel_spectrogram'].to(self.device)
                    real_B = batch_B['mel_spectrogram'].to(self.device)
                    
                    if real_A.size(0) != real_B.size(0):
                        continue
                    
                    # Train Generators
                    optimizers['G'].zero_grad()
                    
                    fake_B = model.G_AB(real_A)
                    fake_A = model.G_BA(real_B)
                    cycle_A = model.G_BA(fake_B)
                    cycle_B = model.G_AB(fake_A)
                    identity_A = model.G_BA(real_A)
                    identity_B = model.G_AB(real_B)
                    
                    D_A_fake = model.D_A(fake_A)
                    D_B_fake = model.D_B(fake_B)
                    
                    gen_output = {
                        'fake_A': fake_A, 'fake_B': fake_B,
                        'cycle_A': cycle_A, 'cycle_B': cycle_B,
                        'identity_A': identity_A, 'identity_B': identity_B,
                        'real_A': real_A, 'real_B': real_B,
                        'D_A_fake': D_A_fake, 'D_B_fake': D_B_fake
                    }
                    
                    generator_losses = criterion.compute_generator_loss(gen_output)
                    generator_losses['total'].backward()
                    optimizers['G'].step()
                    
                    # Train Discriminator A
                    optimizers['D_A'].zero_grad()
                    
                    with torch.no_grad():
                        fake_A_detached = model.G_BA(real_B)
                    
                    D_A_real = model.D_A(real_A)
                    D_A_fake_detached = model.D_A(fake_A_detached)
                    
                    real_loss_A = criterion.adversarial_loss(D_A_real, is_real=True, is_discriminator=True)
                    fake_loss_A = criterion.adversarial_loss(D_A_fake_detached, is_real=False, is_discriminator=True)
                    disc_loss_A = (real_loss_A + fake_loss_A) * 0.5
                    
                    disc_loss_A.backward()
                    optimizers['D_A'].step()
                    
                    # Train Discriminator B
                    optimizers['D_B'].zero_grad()
                    
                    with torch.no_grad():
                        fake_B_detached = model.G_AB(real_A)
                    
                    D_B_real = model.D_B(real_B)
                    D_B_fake_detached = model.D_B(fake_B_detached)
                    
                    real_loss_B = criterion.adversarial_loss(D_B_real, is_real=True, is_discriminator=True)
                    fake_loss_B = criterion.adversarial_loss(D_B_fake_detached, is_real=False, is_discriminator=True)
                    disc_loss_B = (real_loss_B + fake_loss_B) * 0.5
                    
                    disc_loss_B.backward()
                    optimizers['D_B'].step()
                    
                    # Update epoch losses
                    epoch_losses['generator'] += generator_losses['total'].item()
                    epoch_losses['discriminator_A'] += disc_loss_A.item()
                    epoch_losses['discriminator_B'] += disc_loss_B.item()
                    epoch_losses['cycle'] += generator_losses['cycle'].item()
                    epoch_losses['identity'] += generator_losses['identity'].item()
                    
                except Exception as e:
                    print(f"    Batch {batch_idx} failed: {e}")
                    continue
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
            
            # Update history
            history['generator_loss'].append(epoch_losses['generator'])
            history['discriminator_A_loss'].append(epoch_losses['discriminator_A'])
            history['discriminator_B_loss'].append(epoch_losses['discriminator_B'])
            history['cycle_loss'].append(epoch_losses['cycle'])
            history['identity_loss'].append(epoch_losses['identity'])
            
            # Update learning rates
            for scheduler in schedulers.values():
                scheduler.step()
            
            # Log to tensorboard
            global_step = epoch
            self.writer.add_scalar(f'{phase_name}/Generator_Loss', epoch_losses['generator'], global_step)
            self.writer.add_scalar(f'{phase_name}/Discriminator_A_Loss', epoch_losses['discriminator_A'], global_step)
            self.writer.add_scalar(f'{phase_name}/Discriminator_B_Loss', epoch_losses['discriminator_B'], global_step)
            self.writer.add_scalar(f'{phase_name}/Cycle_Loss', epoch_losses['cycle'], global_step)
            self.writer.add_scalar(f'{phase_name}/Identity_Loss', epoch_losses['identity'], global_step)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s) - "
                  f"G: {epoch_losses['generator']:.4f}, "
                  f"D_A: {epoch_losses['discriminator_A']:.4f}, "
                  f"D_B: {epoch_losses['discriminator_B']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{self.experiment_dir}/checkpoints/{phase_name}_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizers_state_dict': {k: v.state_dict() for k, v in optimizers.items()},
                    'schedulers_state_dict': {k: v.state_dict() for k, v in schedulers.items()},
                    'history': history
                }, checkpoint_path)
                print(f"    Saved checkpoint: {checkpoint_path}")
        
        return history
    
    def run_full_training(self, max_files_per_genre: int = None):
        """
        Run the complete training strategy.
        
        Args:
            max_files_per_genre: Maximum files per genre (for testing)
        """
        print(f"\n{'='*70}")
        print(f"STARTING FULL TRAINING STRATEGY: {self.experiment_name}")
        print(f"{'='*70}")
        
        # Setup components
        data_loaders = self.setup_datasets(max_files_per_genre)
        models = self.setup_models()
        optimizers = self.setup_optimizers(models)
        schedulers = self.setup_schedulers(optimizers)
        criteria = self.setup_criteria()
        
        training_results = {}
        
        # Phase 1: Individual pairs
        for phase_name in ['phase1_folk_rock', 'phase1_folk_jazz']:
            print(f"\n{'='*50}")
            print(f"PHASE 1: {self.phases[phase_name]['description']}")
            print(f"{'='*50}")
            
            history = self.train_phase1_cyclegan(
                phase_name=phase_name,
                model=models[phase_name],
                data_loader=data_loaders[phase_name],
                optimizers=optimizers[phase_name],
                schedulers=schedulers[phase_name],
                criterion=criteria[phase_name],
                n_epochs=self.phases[phase_name]['epochs']
            )
            
            training_results[phase_name] = history
        
        # Phase 2: Joint training (placeholder for now)
        print(f"\n{'='*50}")
        print(f"PHASE 2: {self.phases['phase2_joint']['description']}")
        print(f"{'='*50}")
        print("Phase 2 implementation (StarGAN-VC joint training) - Coming next!")
        
        # Save final results
        results_path = f"{self.experiment_dir}/training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for phase, history in training_results.items():
                json_results[phase] = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Training completed! Results saved to: {results_path}")
        
        # Close tensorboard writer
        self.writer.close()
        
        return training_results

def test_training_strategy():
    """Test the training strategy with minimal setup."""
    print("Testing Training Strategy...")
    
    try:
        # Create training strategy
        strategy = TrainingStrategy(
            data_dir="data",
            experiment_name="test_experiment",
            base_lr=0.0002,
            batch_size=4,  # Small for testing
            device='cpu'  # Use CPU for testing
        )
        
        # Run minimal training
        print("\nRunning minimal training test...")
        results = strategy.run_full_training(max_files_per_genre=2)
        
        print(f"\n✓ Training strategy test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Training strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_training_strategy()