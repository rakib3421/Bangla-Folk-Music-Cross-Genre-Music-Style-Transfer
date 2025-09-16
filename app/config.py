"""
Flask Application Configuration
=============================

Configuration classes for different environments.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # Database/Cache settings
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/1'
    RATELIMIT_DEFAULT = "200 per day, 50 per hour"
    
    # SocketIO settings
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Audio processing settings
    SUPPORTED_FORMATS = ['mp3', 'wav', 'flac', 'm4a', 'ogg']
    SUPPORTED_STYLES = ['rock', 'jazz', 'blend']
    DEFAULT_INTENSITY = 0.7
    
    # Cache settings
    CACHE_TTL_HOURS = 24
    MAX_CACHE_SIZE_GB = 5
    MODEL_CACHE_SIZE = 3
    
    # Processing settings
    MAX_QUEUE_SIZE = 100
    PROCESSING_TIMEOUT = 300  # 5 minutes
    
    @staticmethod
    def init_app(app):
        """Initialize app with configuration."""
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    
    # Use SQLite for development if needed
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev_style_transfer.db'
    
    # More verbose logging
    LOG_LEVEL = 'DEBUG'
    
    # Disable some security features for development
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Use PostgreSQL for production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://username:password@localhost/style_transfer'
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    # Rate limiting - stricter in production
    RATELIMIT_DEFAULT = "100 per day, 25 per hour"
    
    @classmethod
    def init_app(cls, app):
        """Initialize production app."""
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.INFO)
        app.logger.addHandler(syslog_handler)

class TestingConfig(Config):
    """Testing configuration."""
    
    TESTING = True
    DEBUG = True
    
    # Use in-memory SQLite for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Use fake Redis for testing
    REDIS_URL = 'redis://localhost:6379/15'  # Different DB for testing
    
    # Smaller limits for testing
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB for testing
    MAX_CACHE_SIZE_GB = 0.1
    CACHE_TTL_HOURS = 1

class DockerConfig(Config):
    """Docker configuration."""
    
    DEBUG = False
    
    # Use environment variables for Docker
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://redis:6379/0'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://postgres:password@db:5432/style_transfer'
    
    # File paths for Docker volumes
    UPLOAD_FOLDER = '/app/uploads'
    CACHE_DIR = '/app/cache'
    
    @classmethod
    def init_app(cls, app):
        """Initialize Docker app."""
        Config.init_app(app)
        
        # Ensure Docker directories exist
        os.makedirs('/app/uploads', exist_ok=True)
        os.makedirs('/app/cache', exist_ok=True)

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}