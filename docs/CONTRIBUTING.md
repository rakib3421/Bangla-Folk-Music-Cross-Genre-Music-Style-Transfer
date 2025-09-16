# Contributing to Bangla Folk to Rock/Jazz Style Transfer System

Thank you for your interest in contributing to our neural audio style transfer project! This document provides guidelines for contributing to the codebase.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and audio processing
- Familiarity with PyTorch

### Setting Up Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/bangla-folk-style-transfer.git
   cd bangla-folk-style-transfer
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

4. **Verify Installation**
   ```bash
   python -m pytest tests/
   python scripts/demo.py --quick-test
   ```

## Development Environment

### Project Structure

```
src/
├── audio/          # Audio processing modules
├── models/         # Neural network architectures
├── training/       # Training pipelines and strategies
├── evaluation/     # Evaluation metrics and testing
└── interactive/    # Interactive control systems

tests/              # Test suite
scripts/            # Utility scripts and demos
docs/               # Documentation
```

### Configuration

The project uses `dev_config.py` for development configuration. Set environment variables:

```bash
export DEBUG=True
export VERBOSE=True
export CUDA_VISIBLE_DEVICES=0
```

## Code Style

### Python Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Import Organization**: Use `isort` for import sorting
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all classes and functions

### Example Function

```python
def transfer_style(
    audio: np.ndarray,
    target_style: str,
    intensity: float = 1.0,
) -> np.ndarray:
    """Transfer musical style to audio.
    
    Args:
        audio: Input audio signal as numpy array.
        target_style: Target style ('rock' or 'jazz').
        intensity: Style transfer intensity (0.0 to 1.0).
    
    Returns:
        Style-transferred audio signal.
        
    Raises:
        ValueError: If target_style is not supported.
        RuntimeError: If model inference fails.
    """
    # Implementation here
    pass
```

### Code Formatting

Use the following tools for code formatting:

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Check style
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

## Testing

### Test Organization

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test module interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test speed and memory usage

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_audio.py -v
pytest tests/phase6_integration_test.py

# Run performance tests
pytest tests/ -m performance
```

### Writing Tests

Example test structure:

```python
import pytest
import numpy as np
from src.audio.audio_preprocessing import AudioPreprocessor


class TestAudioPreprocessor:
    """Test suite for AudioPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create AudioPreprocessor instance for testing."""
        return AudioPreprocessor(sample_rate=22050)
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing."""
        return np.random.randn(22050 * 3)  # 3 seconds
    
    def test_load_audio_success(self, preprocessor, tmp_path):
        """Test successful audio loading."""
        # Test implementation
        pass
    
    def test_preprocess_valid_input(self, preprocessor, sample_audio):
        """Test preprocessing with valid input."""
        result = preprocessor.preprocess(sample_audio)
        assert isinstance(result, dict)
        assert 'features' in result
    
    @pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
    def test_different_sample_rates(self, sample_rate):
        """Test preprocessing with different sample rates."""
        # Test implementation
        pass
```

## Submitting Changes

### Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new style interpolation feature"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(audio): add vocal preservation in style transfer
fix(training): resolve memory leak in batch processing
docs(api): update README with new installation instructions
```

### Pull Request Guidelines

- **Title**: Use conventional commit format
- **Description**: Explain what changes were made and why
- **Tests**: Ensure all tests pass and add new tests if needed
- **Documentation**: Update relevant documentation
- **Breaking Changes**: Clearly document any breaking changes

#### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes don't break existing functionality
```

## Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 1.10.0]
- CUDA version (if applicable): [e.g., 11.3]

**Additional Context**
Any other relevant information.
```

### Performance Issues

For performance-related issues, include:

- Profiling results
- Memory usage patterns
- System specifications
- Test data characteristics

## Feature Requests

### Enhancement Proposals

For new features, create an issue with:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: Other approaches considered
4. **Implementation Notes**: Technical considerations
5. **Testing Strategy**: How to validate the feature

### Research Contributions

For research-related contributions:

1. **Literature Review**: Relevant papers and techniques
2. **Methodology**: Proposed approach
3. **Evaluation Plan**: How to measure success
4. **Experimental Results**: Preliminary findings
5. **Code Integration**: How it fits into existing codebase

## Development Guidelines

### Audio Processing

- Use `librosa` for standard audio processing
- Maintain consistent sample rates (22050 Hz default)
- Handle edge cases (silence, noise, different formats)
- Preserve audio quality throughout pipeline

### Machine Learning

- Use PyTorch for all deep learning components
- Implement proper gradient flow and backpropagation
- Add regularization and normalization layers
- Monitor training metrics and losses

### Performance Optimization

- Profile code before optimizing
- Use appropriate data types (float32 vs float64)
- Implement batch processing for efficiency
- Consider memory usage patterns

### Documentation

- Update API documentation for public functions
- Add examples for complex functionality
- Include performance characteristics
- Document configuration options

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different perspectives and experiences

### Communication

- Use clear and descriptive language
- Provide context for discussions
- Ask questions when uncertain
- Share knowledge and resources

## Resources

### Learning Materials

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Librosa Documentation](https://librosa.org/doc/)
- [Audio Processing Fundamentals](https://www.coursera.org/learn/audio-signal-processing)
- [Deep Learning for Audio](https://arxiv.org/abs/1905.00078)

### Development Tools

- **IDE**: VS Code with Python extension
- **Version Control**: Git with conventional commits
- **Testing**: pytest with coverage reporting
- **Formatting**: Black, isort, flake8
- **Documentation**: Sphinx for API docs

Thank you for contributing to the Bangla Folk to Rock/Jazz Style Transfer System!