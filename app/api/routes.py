"""
RESTful API Routes for Style Transfer System
==========================================

Provides API endpoints for:
- File upload and processing
- Task status checking
- Result download
- User authentication
- Batch processing
"""

import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from functools import wraps
import time

# Import authentication and cache managers
from app.auth import AuthManager
from app.cache import CacheManager

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize managers (these will be injected by the main app)
auth_manager = None
cache_manager = None

def init_api_managers(auth_mgr, cache_mgr):
    """Initialize API managers."""
    global auth_manager, cache_manager
    auth_manager = auth_mgr
    cache_manager = cache_mgr

def require_api_key(f):
    """Decorator to require valid API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        username = auth_manager.validate_api_key(api_key)
        if not username:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Check API quota
        if not auth_manager.check_api_quota(username):
            return jsonify({'error': 'API quota exceeded'}), 429
        
        # Increment usage
        auth_manager.increment_api_usage(username)
        
        # Add username to request context
        request.username = username
        
        return f(*args, **kwargs)
    
    return decorated_function

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api_bp.route('/info', methods=['GET'])
def get_api_info():
    """Get API information and capabilities."""
    return jsonify({
        'name': 'Bangla Folk Style Transfer API',
        'version': '1.0.0',
        'description': 'Transform Bangla folk music to Rock and Jazz styles',
        'supported_formats': ['mp3', 'wav', 'flac', 'm4a', 'ogg'],
        'max_file_size_mb': 50,
        'supported_styles': ['rock', 'jazz', 'blend'],
        'rate_limits': {
            'default': '200 per day, 50 per hour',
            'upload': '10 per minute'
        },
        'endpoints': {
            'upload': 'POST /api/v1/upload',
            'status': 'GET /api/v1/status/{task_id}',
            'download': 'GET /api/v1/download/{task_id}',
            'batch': 'POST /api/v1/batch',
            'auth': {
                'register': 'POST /api/v1/auth/register',
                'login': 'POST /api/v1/auth/login',
                'generate_key': 'POST /api/v1/auth/generate-key'
            }
        }
    })

@api_bp.route('/upload', methods=['POST'])
@require_api_key
def api_upload():
    """API endpoint for file upload and style transfer."""
    try:
        # Check if file is present
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'File type not supported',
                'supported_formats': ['mp3', 'wav', 'flac', 'm4a', 'ogg']
            }), 400
        
        # Get parameters
        target_style = request.form.get('target_style', 'rock')
        if target_style not in ['rock', 'jazz', 'blend']:
            return jsonify({
                'error': 'Invalid target style',
                'supported_styles': ['rock', 'jazz', 'blend']
            }), 400
        
        try:
            intensity = float(request.form.get('intensity', 0.7))
            if not 0.1 <= intensity <= 1.0:
                raise ValueError("Intensity must be between 0.1 and 1.0")
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Additional parameters
        additional_params = {}
        if 'preserve_vocals' in request.form:
            additional_params['preserve_vocals'] = request.form.get('preserve_vocals').lower() == 'true'
        if 'tempo_adjustment' in request.form:
            try:
                additional_params['tempo_adjustment'] = float(request.form.get('tempo_adjustment'))
            except ValueError:
                pass
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        filepath = os.path.join(upload_folder, f"{task_id}_{filename}")
        file.save(filepath)
        
        # Check cache
        cache_key = cache_manager.generate_cache_key(
            filepath, target_style, intensity, additional_params
        )
        cached_result = cache_manager.get_cached_result(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for API task {task_id}")
            return jsonify({
                'task_id': task_id,
                'status': 'completed',
                'message': 'Processing completed (cached result)',
                'result_available': True,
                'download_url': f'/api/v1/download/{task_id}',
                'cached': True,
                'processing_time': 0
            })
        
        # Queue for processing
        from app import processing_queue
        task_data = {
            'task_id': task_id,
            'filepath': filepath,
            'target_style': target_style,
            'intensity': intensity,
            'additional_params': additional_params,
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat(),
            'username': request.username,
            'api_request': True
        }
        
        processing_queue.put(task_data)
        
        # Update task status
        cache_manager.update_task_status(task_id, 'queued', 'Task queued for processing')
        
        return jsonify({
            'task_id': task_id,
            'status': 'queued',
            'message': 'File uploaded successfully, processing queued',
            'estimated_time': '30-120 seconds',
            'status_url': f'/api/v1/status/{task_id}',
            'download_url': f'/api/v1/download/{task_id}'
        }), 202
        
    except Exception as e:
        logger.error(f"API upload error: {str(e)}")
        return jsonify({'error': 'Upload processing failed'}), 500

@api_bp.route('/status/<task_id>', methods=['GET'])
@require_api_key
def api_get_status(task_id):
    """Get processing status for a task."""
    try:
        status_info = cache_manager.get_task_status(task_id)
        
        # Add additional information for API response
        response = {
            'task_id': task_id,
            'status': status_info.get('status', 'not_found'),
            'message': status_info.get('message', 'Task not found'),
            'updated_at': status_info.get('updated_at'),
        }
        
        if status_info.get('status') == 'completed':
            response.update({
                'result_available': True,
                'download_url': f'/api/v1/download/{task_id}'
            })
        elif status_info.get('status') == 'processing':
            response.update({
                'result_available': False,
                'estimated_remaining': '30-90 seconds'
            })
        elif status_info.get('status') == 'failed':
            response.update({
                'result_available': False,
                'error_details': status_info.get('message')
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API status check error: {str(e)}")
        return jsonify({'error': 'Status check failed'}), 500

@api_bp.route('/download/<task_id>', methods=['GET'])
@require_api_key
def api_download(task_id):
    """Download processed audio file."""
    try:
        result_path = cache_manager.get_task_result(task_id)
        
        if not result_path or not os.path.exists(result_path):
            return jsonify({'error': 'Result not found or not ready'}), 404
        
        # Get original filename from task data
        task_status = cache_manager.get_task_status(task_id)
        original_filename = f"transformed_{task_id}.wav"
        
        return send_file(
            result_path,
            as_attachment=True,
            download_name=original_filename,
            mimetype='audio/wav'
        )
        
    except Exception as e:
        logger.error(f"API download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

@api_bp.route('/batch', methods=['POST'])
@require_api_key
def api_batch_upload():
    """Batch processing endpoint for multiple files."""
    try:
        if 'audio_files' not in request.files:
            return jsonify({'error': 'No audio files provided'}), 400
        
        files = request.files.getlist('audio_files')
        if len(files) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 files per batch'}), 400
        
        # Get common parameters
        target_style = request.form.get('target_style', 'rock')
        intensity = float(request.form.get('intensity', 0.7))
        
        batch_id = str(uuid.uuid4())
        task_ids = []
        
        for i, file in enumerate(files):
            if file.filename == '' or not allowed_file(file.filename):
                continue
            
            # Generate task ID for each file
            task_id = f"{batch_id}_{i}"
            task_ids.append(task_id)
            
            # Save file
            filename = secure_filename(file.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            filepath = os.path.join(upload_folder, f"{task_id}_{filename}")
            file.save(filepath)
            
            # Queue for processing
            from app import processing_queue
            task_data = {
                'task_id': task_id,
                'batch_id': batch_id,
                'filepath': filepath,
                'target_style': target_style,
                'intensity': intensity,
                'cache_key': cache_manager.generate_cache_key(filepath, target_style, intensity),
                'timestamp': datetime.now().isoformat(),
                'username': request.username,
                'api_request': True,
                'batch_index': i
            }
            
            processing_queue.put(task_data)
            cache_manager.update_task_status(task_id, 'queued', f'Batch task {i+1} queued')
        
        return jsonify({
            'batch_id': batch_id,
            'task_ids': task_ids,
            'status': 'queued',
            'message': f'{len(task_ids)} files queued for processing',
            'batch_status_url': f'/api/v1/batch/{batch_id}/status'
        }), 202
        
    except Exception as e:
        logger.error(f"API batch upload error: {str(e)}")
        return jsonify({'error': 'Batch upload failed'}), 500

@api_bp.route('/batch/<batch_id>/status', methods=['GET'])
@require_api_key
def api_batch_status(batch_id):
    """Get status for all tasks in a batch."""
    try:
        # Get all tasks for this batch
        batch_tasks = []
        
        # This is a simplified implementation
        # In production, you'd want to store batch metadata
        for i in range(10):  # Check up to 10 tasks
            task_id = f"{batch_id}_{i}"
            status_info = cache_manager.get_task_status(task_id)
            
            if status_info.get('status') != 'not_found':
                batch_tasks.append({
                    'task_id': task_id,
                    'status': status_info.get('status'),
                    'message': status_info.get('message'),
                    'download_url': f'/api/v1/download/{task_id}' if status_info.get('status') == 'completed' else None
                })
        
        # Calculate batch status
        if not batch_tasks:
            return jsonify({'error': 'Batch not found'}), 404
        
        completed_count = sum(1 for task in batch_tasks if task['status'] == 'completed')
        failed_count = sum(1 for task in batch_tasks if task['status'] == 'failed')
        processing_count = sum(1 for task in batch_tasks if task['status'] in ['queued', 'processing'])
        
        batch_status = 'completed' if completed_count == len(batch_tasks) else \
                      'failed' if failed_count == len(batch_tasks) else \
                      'processing'
        
        return jsonify({
            'batch_id': batch_id,
            'status': batch_status,
            'total_tasks': len(batch_tasks),
            'completed': completed_count,
            'failed': failed_count,
            'processing': processing_count,
            'tasks': batch_tasks
        })
        
    except Exception as e:
        logger.error(f"API batch status error: {str(e)}")
        return jsonify({'error': 'Batch status check failed'}), 500

# Authentication endpoints
@api_bp.route('/auth/register', methods=['POST'])
def api_register():
    """Register new user."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        success = auth_manager.register_user(username, password, email)
        
        if success:
            return jsonify({
                'message': 'User registered successfully',
                'username': username
            }), 201
        else:
            return jsonify({'error': 'Username already exists'}), 409
            
    except Exception as e:
        logger.error(f"API register error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@api_bp.route('/auth/login', methods=['POST'])
def api_login():
    """User login."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        if auth_manager.authenticate_user(username, password):
            session_id = auth_manager.create_session(username)
            user_info = auth_manager.get_user_info(username)
            
            return jsonify({
                'message': 'Login successful',
                'session_id': session_id,
                'user': user_info
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        logger.error(f"API login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@api_bp.route('/auth/generate-key', methods=['POST'])
def api_generate_key():
    """Generate API key for authenticated user."""
    try:
        # Check session or basic auth
        session_id = request.headers.get('X-Session-ID')
        username = None
        
        if session_id:
            username = auth_manager.validate_session(session_id)
        else:
            # Try basic auth
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Basic '):
                import base64
                try:
                    credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
                    user, pwd = credentials.split(':', 1)
                    if auth_manager.authenticate_user(user, pwd):
                        username = user
                except:
                    pass
        
        if not username:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json() or {}
        description = data.get('description', 'API key generated via API')
        
        api_key = auth_manager.generate_api_key(username, description)
        
        return jsonify({
            'api_key': api_key,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'usage': 'Include this key in X-API-Key header for API requests'
        })
        
    except Exception as e:
        logger.error(f"API key generation error: {str(e)}")
        return jsonify({'error': 'API key generation failed'}), 500

@api_bp.route('/auth/profile', methods=['GET'])
@require_api_key
def api_get_profile():
    """Get user profile information."""
    try:
        user_info = auth_manager.get_user_info(request.username)
        api_keys = auth_manager.get_user_api_keys(request.username)
        
        return jsonify({
            'user': user_info,
            'api_keys': api_keys
        })
        
    except Exception as e:
        logger.error(f"API profile error: {str(e)}")
        return jsonify({'error': 'Failed to get profile'}), 500

@api_bp.route('/cache/stats', methods=['GET'])
@require_api_key
def api_cache_stats():
    """Get cache statistics."""
    try:
        stats = cache_manager.get_cache_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"API cache stats error: {str(e)}")
        return jsonify({'error': 'Failed to get cache stats'}), 500

# Error handlers
@api_bp.errorhandler(404)
def api_not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@api_bp.errorhandler(405)
def api_method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@api_bp.errorhandler(413)
def api_payload_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@api_bp.errorhandler(429)
def api_rate_limit_exceeded(error):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429

@api_bp.errorhandler(500)
def api_internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500