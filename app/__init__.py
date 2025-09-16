"""
Phase 7: Deployment and Interface - Flask Web Application
========================================================

Main Flask application for Bangla Folk to Rock/Jazz Style Transfer
Features:
- Web Interface for file upload and style selection
- RESTful API for batch processing
- WebSocket for real-time streaming
- Authentication and rate limiting
- Caching strategy for performance optimization
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import redis
from datetime import datetime
import uuid
import threading
import queue

# Import our style transfer modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.advanced_style_transfer import AdvancedStyleTransfer
from src.audio.audio_preprocessing import AudioPreprocessor
from src.interactive.real_time_controller import RealTimeController
from app.auth.authentication import AuthManager
from app.cache.cache_manager import CacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    app.config['RATE_LIMIT_STORAGE_URL'] = app.config['REDIS_URL']
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize extensions
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize rate limiter
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Initialize components
    auth_manager = AuthManager()
    cache_manager = CacheManager(app.config['REDIS_URL'])
    
    # Global processing queue
    processing_queue = queue.Queue()
    
    @app.route('/')
    def index():
        """Main page with file upload interface."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    @limiter.limit("10 per minute")
    def upload_file():
        """Handle file upload and initiate style transfer."""
        try:
            if 'audio_file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['audio_file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Generate unique task ID
            task_id = str(uuid.uuid4())
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
            file.save(filepath)
            
            # Get style selection
            target_style = request.form.get('target_style', 'rock')
            intensity = float(request.form.get('intensity', 0.7))
            
            # Check cache first
            cache_key = cache_manager.generate_cache_key(filepath, target_style, intensity)
            cached_result = cache_manager.get_cached_result(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for task {task_id}")
                return jsonify({
                    'task_id': task_id,
                    'status': 'completed',
                    'result_url': cached_result,
                    'cached': True
                })
            
            # Queue for processing
            task_data = {
                'task_id': task_id,
                'filepath': filepath,
                'target_style': target_style,
                'intensity': intensity,
                'cache_key': cache_key,
                'timestamp': datetime.now().isoformat()
            }
            
            processing_queue.put(task_data)
            
            return jsonify({
                'task_id': task_id,
                'status': 'queued',
                'message': 'File uploaded successfully, processing started'
            })
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return jsonify({'error': 'Upload failed'}), 500
    
    @app.route('/status/<task_id>')
    def get_status(task_id):
        """Get processing status for a task."""
        try:
            status = cache_manager.get_task_status(task_id)
            return jsonify(status)
        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return jsonify({'error': 'Status check failed'}), 500
    
    @app.route('/download/<task_id>')
    def download_result(task_id):
        """Download processed audio file."""
        try:
            result_path = cache_manager.get_task_result(task_id)
            if result_path and os.path.exists(result_path):
                return send_file(result_path, as_attachment=True)
            else:
                return jsonify({'error': 'File not found'}), 404
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return jsonify({'error': 'Download failed'}), 500
    
    @socketio.on('connect')
    def handle_connect():
        """Handle WebSocket connection."""
        logger.info(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to style transfer service'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe_task')
    def handle_task_subscription(data):
        """Subscribe to task updates."""
        task_id = data.get('task_id')
        if task_id:
            # Join room for this task
            from flask_socketio import join_room
            join_room(f"task_{task_id}")
            emit('subscribed', {'task_id': task_id})
    
    def allowed_file(filename):
        """Check if file extension is allowed."""
        ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Start background processing thread
    def background_processor():
        """Background thread for processing audio files."""
        style_transfer = AdvancedStyleTransfer()
        preprocessor = AudioPreprocessor()
        
        while True:
            try:
                task_data = processing_queue.get(timeout=1)
                process_audio_task(task_data, style_transfer, preprocessor, socketio, cache_manager)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Background processor error: {str(e)}")
    
    # Start background thread
    processor_thread = threading.Thread(target=background_processor, daemon=True)
    processor_thread.start()
    
    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    return app, socketio

def process_audio_task(task_data, style_transfer, preprocessor, socketio, cache_manager):
    """Process audio style transfer task."""
    task_id = task_data['task_id']
    
    try:
        # Update status
        cache_manager.update_task_status(task_id, 'processing', 'Starting style transfer')
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'processing',
            'progress': 0,
            'message': 'Loading audio file...'
        }, room=f"task_{task_id}")
        
        # Load and preprocess audio
        audio_data, sr = preprocessor.load_audio(task_data['filepath'])
        
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'processing',
            'progress': 20,
            'message': 'Analyzing audio features...'
        }, room=f"task_{task_id}")
        
        # Perform style transfer
        result_audio = style_transfer.transfer_style(
            audio_data,
            target_style=task_data['target_style'],
            intensity=task_data['intensity']
        )
        
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'processing',
            'progress': 80,
            'message': 'Saving result...'
        }, room=f"task_{task_id}")
        
        # Save result
        output_filename = f"{task_id}_result.wav"
        output_path = os.path.join(os.path.dirname(task_data['filepath']), output_filename)
        preprocessor.save_audio(result_audio, output_path, sr)
        
        # Cache result
        cache_manager.cache_result(task_data['cache_key'], output_path)
        cache_manager.update_task_status(task_id, 'completed', 'Style transfer completed', output_path)
        
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'completed',
            'progress': 100,
            'message': 'Style transfer completed!',
            'download_url': f'/download/{task_id}'
        }, room=f"task_{task_id}")
        
    except Exception as e:
        logger.error(f"Task processing error: {str(e)}")
        cache_manager.update_task_status(task_id, 'failed', str(e))
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'failed',
            'message': f'Processing failed: {str(e)}'
        }, room=f"task_{task_id}")

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)