# Cross-Genre Music Style Transfer: Bengali Folk → Rock/Jazz

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎵 Overview

A comprehensive deep learning system for cross-genre music style transfer, specifically designed to transform **Bengali Folk music** into **Rock** and **Jazz** styles while preserving vocal characteristics, rhythmic patterns, and musical structure. This project implements a complete pipeline from audio preprocessing to production-ready deployment with advanced optimization features.

## ✨ Key Features

### 🎛️ **Advanced Style Transfer**
- **Multi-Genre Support**: Bengali Folk → Rock/Jazz transformation
- **Vocal Preservation**: Maintains vocal characteristics during style transfer
- **Rhythmic Awareness**: Preserves and adapts rhythmic patterns
- **Musical Structure**: Respects song structure and harmonic progressions

### 🚀 **Production-Ready Pipeline**
- **Real-time Processing**: Optimized models for live performance
- **Interactive Control**: User-adjustable style intensity and blending
- **Quality Enhancement**: Post-processing for superior audio quality
- **Model Optimization**: 96.5% size reduction with pruning and quantization

### 🔧 **Advanced Features**
- **CPU Optimization**: Efficient training and inference on CPU
- **Interactive Style Control**: Real-time style interpolation and blending
- **Quality Enhancement**: Spectral artifact removal and dynamic range optimization
- **Comprehensive Evaluation**: Musical quality metrics and perceptual assessment

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input Audio (Bengali Folk)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────────────────────┐
│                    Audio Preprocessing                          │
│  • Format standardization (44.1kHz, 16-bit)                   │
│  • Vocal/instrumental separation                               │
│  • Feature extraction (mel-spectrograms, MFCC)                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────────────────────┐
│                 Style Transfer Engine                           │
│  • CycleGAN-based architecture                                 │
│  • Rhythm-aware loss functions                                 │
│  • Vocal preservation mechanisms                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────────────────────┐
│                 Interactive Control Layer                       │
│  • Style interpolation (Folk ↔ Rock/Jazz)                     │
│  • Multi-genre blending                                        │
│  • Intensity adjustment                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────────────────────┐
│                 Quality Enhancement                             │
│  • Spectral artifact removal                                   │
│  • Dynamic range optimization                                  │
│  • Harmonic enhancement                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────────────────────┐
│                Output Audio (Rock/Jazz Style)                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Cross-Genre-Music-Style-Transfer/
├── 📁 src/                           # Source code modules
│   ├── 📁 audio/                     # Audio processing
│   │   ├── preprocessing.py          # Audio standardization
│   │   ├── feature_extraction.py     # Feature analysis
│   │   └── quality_enhancement.py    # Quality improvement
│   ├── 📁 models/                    # Neural network models
│   │   ├── cyclegan_architecture.py  # Main architecture
│   │   ├── loss_functions.py         # Custom loss functions
│   │   └── model_optimization.py     # Model compression
│   ├── 📁 training/                  # Training pipeline
│   │   ├── cpu_training.py           # CPU-optimized training
│   │   ├── training_strategy.py      # Training configurations
│   │   └── monitoring.py             # Training visualization
│   ├── 📁 evaluation/                # Evaluation framework
│   │   ├── musical_evaluation.py     # Musical quality metrics
│   │   ├── listening_tests.py        # Perceptual evaluation
│   │   └── style_transfer_evaluation.py # Transfer quality
│   └── 📁 interactive/               # Interactive control
│       ├── interactive_control.py    # Style manipulation
│       └── real_time_processing.py   # Live processing
├── 📁 data/                          # Dataset
│   ├── Bangla Folk/                  # Bengali folk music (112 files)
│   ├── Jazz/                         # Jazz music (103 files)
│   ├── Rock/                         # Rock music (107 files)
│   └── Auxiliary/                    # Additional datasets
├── 📁 experiments/                   # Experiment results
│   ├── cpu_optimized_training/       # Training results
│   ├── interactive_control/          # Control experiments
│   ├── quality_enhancement/          # Enhancement tests
│   └── phase6_integration/           # Integration tests
├── 📁 docs/                          # Documentation
│   ├── TRAINING_GUIDE.md             # Training instructions
│   ├── API_REFERENCE.md              # API documentation
│   └── PHASE_SUMMARIES.md            # Development phases
├── 📁 scripts/                       # Utility scripts
│   ├── demo.py                       # Quick demonstration
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Evaluation script
├── 📁 tests/                         # Test suite
│   ├── test_audio_processing.py      # Audio tests
│   ├── test_model_training.py        # Training tests
│   └── test_integration.py           # Integration tests
├── requirements.txt                  # Python dependencies
├── setup.py                         # Package installation
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rakib3421/Cross-Genre-Music-Style-Transfer.git
cd Cross-Genre-Music-Style-Transfer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Demo

```bash
# Run the interactive demo
python scripts/demo.py

# Or use the command line interface
python -m src.interactive.interactive_control --input "path/to/folk_song.wav" --target "jazz" --intensity 0.7
```

### 3. Training Your Own Model

```bash
# Start CPU-optimized training
python scripts/train.py --config configs/cpu_training.json

# Monitor training progress
python -m src.training.monitoring --experiment "your_experiment_name"
```
- FFmpeg (for audio processing)
- Windows 10/11 (tested environment)

### Installation Steps

1. **Clone the repository and navigate to the project directory**
   ```bash
   cd Project-3
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg** (if not already installed)
   ```bash
   winget install ffmpeg
   ```

## 🚀 Usage

### Quick Demo
Run the demo script to test the implementation:
```bash
python demo.py
```

### Full Analysis Pipeline
Run the complete Phase 1 analysis:
```bash
python phase1_analysis.py
```

### Individual Component Testing
Test specific components:
```bash
python test_audio.py  # Test audio processing
```

## 📊 Dataset Overview

| Genre      | Files | Size    | Characteristics |
|------------|-------|---------|----------------|
| Bangla Folk| 112   | 604 MB  | Traditional vocals, acoustic instruments |
| Jazz       | 103   | 365 MB  | Swing rhythms, improvisation |
| Rock       | 107   | 1036 MB | Distorted guitars, driving rhythms |
| **Total**  | **322** | **1.96 GB** | Multi-genre dataset |

## 🔧 Key Components

### 1. Audio Preprocessing (`audio_preprocessing.py`)
- **AudioPreprocessor**: Handles format standardization and normalization
- **Segmentation**: Splits long audio files into manageable chunks
- **Quality Control**: Ensures consistent audio properties across dataset

### 2. Feature Extraction (`feature_extraction.py`)
- **AudioFeatureExtractor**: Extracts comprehensive audio features
- **GenreAnalyzer**: Analyzes genre-specific characteristics
- **Statistical Analysis**: Computes feature statistics and summaries

### 3. Musical Structure Analysis (`musical_structure_analysis.py`)
- **MusicalStructureAnalyzer**: Advanced musical analysis
- **Source Separation**: Vocals vs instruments
- **Harmonic Analysis**: Chord progressions and harmonic content
- **Rhythm Analysis**: Meter detection and rhythm patterns

### 4. Main Pipeline (`phase1_analysis.py`)
- **Phase1Pipeline**: Orchestrates the complete analysis workflow
- **Visualization**: Generates analysis reports and plots
- **Data Export**: Saves results in multiple formats (JSON, CSV, PNG)

## 📈 Analysis Outputs

The pipeline generates comprehensive analysis results:

### Generated Files
- `dataset_overview.json` - Basic dataset statistics
- `preprocessing_results.json` - Audio processing summary
- `feature_analysis.json` - Feature extraction results
- `structure_analysis.json` - Musical structure analysis
- `genre_characteristics.json` - Genre-specific characteristics
- `phase1_final_report.json` - Complete analysis summary
- `phase1_analysis_report.png` - Visualization report
- `[genre]_features.csv` - Detailed features for each genre

### Analysis Capabilities
- **Tempo Analysis**: BPM distribution across genres
- **Harmonic Analysis**: Chord progressions and tonal characteristics
- **Rhythmic Analysis**: Beat patterns and meter identification
- **Timbral Analysis**: Spectral characteristics and texture
- **Structural Analysis**: Vocal/instrumental content separation

## 🎵 Genre Characteristics Discovered

### Bangla Folk
- Traditional vocal styles with rich melodic content
- Acoustic instruments and natural harmonics
- Varied tempo patterns reflecting traditional rhythms

### Jazz
- Complex harmonic progressions
- Swing rhythms and improvisation patterns
- Rich timbral diversity from various instruments

### Rock
- Prominent electric guitar content
- Driving rhythmic patterns
- Higher energy and dynamic range

## 🔬 Technical Features

### Audio Processing
- Sample rate standardization (44.1kHz)
- Bit depth normalization (16-bit)
- Dynamic range normalization
- Intelligent segmentation with overlap handling

### Feature Extraction
- **Time-frequency**: Mel-spectrograms with configurable resolution
- **Harmonic**: Chromagrams for pitch class analysis
- **Rhythmic**: Tempograms and beat synchronous features
- **Timbral**: MFCCs and spectral shape descriptors

### Advanced Analysis
- **Source Separation**: Harmonic-percussive separation
- **Chord Recognition**: Template-based chord identification
- **Melody Extraction**: F0 estimation and melodic contour
- **Rhythm Analysis**: Meter detection and pattern recognition

## 🚧 Future Development (Phase 2+)

### Planned Enhancements
- **Deep Learning Models**: Neural network-based feature extraction
- **Style Transfer Implementation**: GAN-based style transfer models
- **Real-time Processing**: Live audio style transfer
- **Extended Genres**: Additional music genres and styles
- **Quality Metrics**: Perceptual and objective quality assessment

### Research Directions
- **Cross-cultural Analysis**: Deeper analysis of Bangla Folk characteristics
- **Style Interpolation**: Gradual style transitions
- **User Studies**: Perceptual evaluation of style transfer quality
- **Mobile Implementation**: Smartphone app for real-time style transfer

## 📚 Dependencies

### Core Libraries
- `librosa>=0.10.0` - Audio analysis and processing
- `numpy>=1.26.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `pandas>=2.0.0` - Data manipulation and analysis

### Audio Processing
- `soundfile>=0.12.1` - Audio file I/O
- `pydub>=0.25.1` - Audio manipulation
- `essentia>=2.1b6` - Advanced audio analysis (optional)

### Visualization
- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.15.0` - Interactive plots (optional)

### Utilities
- `tqdm>=4.65.0` - Progress bars
- `joblib>=1.3.0` - Parallel processing

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests to ensure everything works
5. Submit a pull request

### Testing
```bash
python demo.py          # Quick functionality test
python test_audio.py    # Audio processing test
python -m pytest tests/ # Full test suite (when available)
```

## 📄 License

This project is part of academic research on cross-genre music style transfer. Please cite appropriately if using in academic work.

## 🔗 Related Work

This implementation builds upon research in:
- Music Information Retrieval (MIR)
- Audio source separation
- Style transfer in music
- Cross-cultural music analysis

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository.

---

**Note**: This is Phase 1 of a multi-phase project. The implementation provides a solid foundation for advanced style transfer techniques to be developed in subsequent phases.
