"""Development configuration and utilities."""

import os
import logging
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
for dir_path in [DATA_DIR, SCRIPTS_DIR, TESTS_DIR, DOCS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Logging configuration
def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('style_transfer.log'),
            logging.StreamHandler()
        ]
    )
    
    # Suppress some noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Environment variables
def load_environment():
    """Load environment variables with defaults."""
    return {
        'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES', '0'),
        'PYTORCH_CUDA_ALLOC_CONF': os.getenv('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512'),
        'WANDB_PROJECT': os.getenv('WANDB_PROJECT', 'bangla-folk-style-transfer'),
        'MODEL_CACHE_DIR': os.getenv('MODEL_CACHE_DIR', str(PROJECT_ROOT / 'models')),
        'RESULTS_DIR': os.getenv('RESULTS_DIR', str(PROJECT_ROOT / 'results')),
    }

# Development utilities
class DevelopmentConfig:
    """Development configuration class."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.src_dir = SRC_DIR
        self.data_dir = DATA_DIR
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.verbose = os.getenv('VERBOSE', 'False').lower() == 'true'
        self.env_vars = load_environment()
        
    def setup(self):
        """Set up development environment."""
        # Set up logging
        level = logging.DEBUG if self.debug else logging.INFO
        logger = setup_logging(level)
        
        if self.verbose:
            logger.info(f"Project root: {self.project_root}")
            logger.info(f"Source directory: {self.src_dir}")
            logger.info(f"Data directory: {self.data_dir}")
            logger.info(f"Debug mode: {self.debug}")
            
        return logger

# Initialize development configuration
dev_config = DevelopmentConfig()
logger = dev_config.setup()

if __name__ == "__main__":
    print("Development environment initialized successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Debug mode: {dev_config.debug}")