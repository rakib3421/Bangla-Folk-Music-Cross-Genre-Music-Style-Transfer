"""
Bangla Folk to Rock/Jazz Style Transfer System

A comprehensive neural audio style transfer system for converting Bangla Folk music
to Rock and Jazz styles while preserving cultural authenticity and musical quality.
"""

__version__ = "1.0.0"
__author__ = "Style Transfer Research Team"
__email__ = "contact@example.com"

# Package imports
from . import audio
from . import models
from . import training
from . import evaluation
from . import interactive

__all__ = [
    "audio",
    "models", 
    "training",
    "evaluation",
    "interactive"
]