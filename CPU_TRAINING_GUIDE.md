# CPU-Optimized Cross-Genre Music Style Transfer

## Quick Start Guide for CPU Training

This guide provides optimized configurations and scripts for training the cross-genre music style transfer system on CPU-only systems.

## CPU Optimizations Implemented

### 1. Model Architecture Optimizations
- **Reduced Model Complexity**: 32 channels (vs 64 for GPU)
- **Fewer Residual Blocks**: 4 blocks (vs 6-9 for GPU)
- **Smaller Input Size**: 64x64 (vs 128x128 for GPU)
- **Reduced Mel Bins**: 64 (vs 128 for GPU)
- **Total Parameters**: ~4.1M (vs ~28M for full model)

### 2. Training Optimizations
- **Small Batch Size**: 2 (vs 16-32 for GPU)
- **Gradient Accumulation**: 4 steps to simulate larger batch
- **Lower Sample Rate**: 16kHz (vs 22kHz for GPU)
- **Shorter Segments**: 2 seconds (vs 5 seconds for GPU)
- **Limited Dataset**: 10 files per genre for testing

### 3. Memory Optimizations
- **CPU Thread Optimization**: Uses available CPU cores efficiently
- **MKL-DNN Backend**: Enabled for CPU acceleration
- **Disabled Multiprocessing**: No worker processes for data loading
- **Memory Efficient Loading**: Reduced prefetching and caching

## Usage

### 1. Basic CPU Training

```python
# Run the CPU-optimized training script
python cpu_training.py
```

### 2. Custom Configuration

```python
from cpu_optimization import CPUOptimizedConfig
from cpu_training import CPUOptimizedTrainer

# Create custom configuration
config = CPUOptimizedConfig()
config.training_config['batch_size'] = 1  # Even smaller batch
config.audio_config['max_files_per_genre'] = 5  # Fewer files

# Create and run trainer
trainer = CPUOptimizedTrainer(
    data_dir="data",
    experiment_name="my_cpu_experiment"
)
results = trainer.run_training()
```

### 3. Test CPU Setup

```python
# Test your CPU configuration
python cpu_optimization.py
```

## Expected Performance

### Training Time Estimates (CPU)
- **Per Epoch**: ~2-5 minutes (depends on CPU)
- **Phase 1 (10 epochs)**: ~30-50 minutes
- **Phase 2 (10 epochs)**: ~30-50 minutes
- **Total Training**: ~1-2 hours

### Memory Usage
- **Model Size**: ~15.7 MB
- **Peak RAM**: ~2-4 GB (depending on batch size)
- **Disk Space**: ~500 MB for experiments

### Quality Expectations
- **Reduced Quality**: Due to smaller model and shorter training
- **Good for**: Proof of concept, experimentation, learning
- **Production Use**: Consider cloud GPU for full quality

## Files Created

### Core CPU Scripts
- `cpu_optimization.py`: CPU-specific configurations and optimizations
- `cpu_training.py`: Complete CPU-optimized training pipeline
- `cpu_optimized_config.json`: Saved configuration parameters

### Training Outputs
- `experiments/cpu_optimized_training/`: Main experiment directory
  - `checkpoints/`: Model checkpoints every 10 epochs
  - `logs/`: TensorBoard logs for monitoring
  - `visualizations/`: Training curves and spectrograms
  - `audio_samples/`: Generated audio samples
  - `config.json`: Experiment configuration

## Monitoring Training

### TensorBoard (Recommended)
```bash
# Start TensorBoard (in new terminal)
tensorboard --logdir experiments/cpu_optimized_training/logs

# Open browser to: http://localhost:6006
```

### Console Output
- Real-time loss values
- Epoch timing information
- Checkpoint save notifications
- Error handling messages

## Tips for CPU Training

### 1. System Optimization
- Close unnecessary applications
- Use all available CPU cores
- Ensure adequate RAM (8GB+ recommended)
- Use SSD for faster data loading

### 2. Configuration Tuning
- **Reduce batch size** if running out of memory
- **Increase gradient accumulation** to simulate larger batches
- **Limit dataset size** for faster iteration
- **Reduce model complexity** further if needed

### 3. Monitoring
- Watch for memory usage patterns
- Monitor CPU utilization
- Check loss convergence
- Listen to generated audio samples

## Common Issues & Solutions

### 1. Out of Memory
```python
# Reduce batch size
config.training_config['batch_size'] = 1

# Reduce model size
config.model_config['base_generator_channels'] = 16
```

### 2. Slow Training
```python
# Reduce dataset size
config.audio_config['max_files_per_genre'] = 5

# Shorter segments
config.audio_config['segment_length'] = 1.0
```

### 3. Poor Audio Quality
- Use higher sample rate: `config.audio_config['sample_rate'] = 22050`
- Increase model size: `config.model_config['base_generator_channels'] = 48`
- Train for more epochs: `config.training_config['max_epochs_phase1'] = 50`

## Next Steps

1. **Start with CPU training** to verify the pipeline works
2. **Analyze results** using TensorBoard and audio samples
3. **Tune hyperparameters** based on initial results
4. **Consider cloud GPU** for production training if needed

## Hardware Recommendations

### Minimum Requirements
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **OS**: Windows 10+, macOS 10.15+, Linux

### Optimal Setup
- **CPU**: 8+ cores, 3.0+ GHz (Intel i7/AMD Ryzen 7+)
- **RAM**: 16 GB
- **Storage**: SSD with 20+ GB free space
- **Cooling**: Adequate CPU cooling for sustained load

## Support

For issues or questions:
1. Check console output for error messages
2. Review TensorBoard logs for training progress
3. Verify dataset structure and file formats
4. Test with smaller configuration if problems persist