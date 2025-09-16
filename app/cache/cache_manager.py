"""
Cache Manager for Style Transfer System
=====================================

Implements caching strategy for:
- Model caching for faster inference
- Result caching for repeated requests
- Progressive loading for large files
"""

import os
import hashlib
import pickle
import time
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import redis
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for the style transfer system."""
    
    def __init__(self, redis_url: str = None, cache_dir: str = None):
        """Initialize cache manager."""
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', 'cache_data')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Redis for fast access and task status
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info("Redis cache enabled")
            else:
                self.redis_available = False
                logger.info("Redis not configured, using file-based cache only")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to file cache")
            self.redis_available = False
        
        # Model cache for loaded models
        self.model_cache = {}
        self.model_cache_size = 3  # Max models to keep in memory
        
        # File cache settings
        self.max_cache_size_gb = 5  # Maximum cache size in GB
        self.cache_ttl_hours = 24  # Cache time-to-live in hours
    
    def generate_cache_key(self, filepath: str, target_style: str, intensity: float, 
                          additional_params: Dict = None) -> str:
        """Generate unique cache key for audio processing request."""
        # Get file hash for content-based caching
        file_hash = self._get_file_hash(filepath)
        
        # Create parameter string
        params = f"{target_style}_{intensity}"
        if additional_params:
            params += "_" + "_".join(f"{k}:{v}" for k, v in sorted(additional_params.items()))
        
        # Generate cache key
        cache_key = hashlib.md5(f"{file_hash}_{params}".encode()).hexdigest()
        return cache_key
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def cache_result(self, cache_key: str, result_path: str, metadata: Dict = None):
        """Cache processing result."""
        cache_info = {
            'result_path': result_path,
            'cached_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=self.cache_ttl_hours)).isoformat(),
            'metadata': metadata or {}
        }
        
        # Store in Redis if available
        if self.redis_available:
            try:
                self.redis_client.setex(
                    f"result:{cache_key}",
                    timedelta(hours=self.cache_ttl_hours),
                    json.dumps(cache_info)
                )
            except Exception as e:
                logger.error(f"Redis cache store failed: {e}")
        
        # Store in file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(cache_info, f, indent=2)
        
        # Clean up old cache files
        self._cleanup_cache()
    
    def get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get cached result if available and not expired."""
        # Try Redis first
        if self.redis_available:
            try:
                cached_data = self.redis_client.get(f"result:{cache_key}")
                if cached_data:
                    cache_info = json.loads(cached_data)
                    result_path = cache_info['result_path']
                    if os.path.exists(result_path):
                        return result_path
            except Exception as e:
                logger.error(f"Redis cache read failed: {e}")
        
        # Try file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_info = json.load(f)
                
                # Check if expired
                expires_at = datetime.fromisoformat(cache_info['expires_at'])
                if datetime.now() > expires_at:
                    os.remove(cache_file)
                    return None
                
                result_path = cache_info['result_path']
                if os.path.exists(result_path):
                    return result_path
                
            except Exception as e:
                logger.error(f"File cache read failed: {e}")
        
        return None
    
    def cache_model(self, model_name: str, model_instance: Any):
        """Cache model instance in memory."""
        # Remove oldest model if cache is full
        if len(self.model_cache) >= self.model_cache_size:
            oldest_key = min(self.model_cache.keys(), 
                           key=lambda k: self.model_cache[k]['last_used'])
            del self.model_cache[oldest_key]
            logger.info(f"Removed cached model: {oldest_key}")
        
        self.model_cache[model_name] = {
            'instance': model_instance,
            'cached_at': time.time(),
            'last_used': time.time()
        }
        logger.info(f"Cached model: {model_name}")
    
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get cached model instance."""
        if model_name in self.model_cache:
            self.model_cache[model_name]['last_used'] = time.time()
            return self.model_cache[model_name]['instance']
        return None
    
    def update_task_status(self, task_id: str, status: str, message: str = None, 
                          result_path: str = None):
        """Update task processing status."""
        task_info = {
            'task_id': task_id,
            'status': status,
            'message': message,
            'result_path': result_path,
            'updated_at': datetime.now().isoformat()
        }
        
        # Store in Redis if available
        if self.redis_available:
            try:
                self.redis_client.setex(
                    f"task:{task_id}",
                    timedelta(hours=24),  # Task status TTL
                    json.dumps(task_info)
                )
            except Exception as e:
                logger.error(f"Redis task status update failed: {e}")
        
        # Store in file cache as backup
        task_file = os.path.join(self.cache_dir, f"task_{task_id}.json")
        with open(task_file, 'w') as f:
            json.dump(task_info, f, indent=2)
    
    def get_task_status(self, task_id: str) -> Dict:
        """Get task processing status."""
        # Try Redis first
        if self.redis_available:
            try:
                task_data = self.redis_client.get(f"task:{task_id}")
                if task_data:
                    return json.loads(task_data)
            except Exception as e:
                logger.error(f"Redis task status read failed: {e}")
        
        # Try file cache
        task_file = os.path.join(self.cache_dir, f"task_{task_id}.json")
        if os.path.exists(task_file):
            try:
                with open(task_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"File task status read failed: {e}")
        
        return {'task_id': task_id, 'status': 'not_found', 'message': 'Task not found'}
    
    def get_task_result(self, task_id: str) -> Optional[str]:
        """Get task result file path."""
        task_status = self.get_task_status(task_id)
        if task_status.get('status') == 'completed':
            result_path = task_status.get('result_path')
            if result_path and os.path.exists(result_path):
                return result_path
        return None
    
    def _cleanup_cache(self):
        """Clean up old cache files."""
        try:
            current_time = datetime.now()
            total_size = 0
            cache_files = []
            
            # Collect cache files info
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    stat = os.stat(filepath)
                    cache_files.append({
                        'path': filepath,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
                    total_size += stat.st_size
            
            # Check if cleanup is needed
            max_size_bytes = self.max_cache_size_gb * 1024 * 1024 * 1024
            if total_size > max_size_bytes:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x['modified'])
                
                # Remove files until under size limit
                for file_info in cache_files:
                    if total_size <= max_size_bytes:
                        break
                    
                    try:
                        os.remove(file_info['path'])
                        total_size -= file_info['size']
                        logger.info(f"Removed old cache file: {file_info['path']}")
                        
                        # Also remove associated result file if exists
                        cache_key = os.path.splitext(os.path.basename(file_info['path']))[0]
                        if cache_key.startswith('task_'):
                            continue
                        
                        # Load cache info to find result file
                        with open(file_info['path'], 'r') as f:
                            cache_info = json.load(f)
                            result_path = cache_info.get('result_path')
                            if result_path and os.path.exists(result_path):
                                os.remove(result_path)
                                
                    except Exception as e:
                        logger.error(f"Error removing cache file: {e}")
            
            # Remove expired files
            for file_info in cache_files:
                if current_time - file_info['modified'] > timedelta(hours=self.cache_ttl_hours):
                    try:
                        os.remove(file_info['path'])
                        logger.info(f"Removed expired cache file: {file_info['path']}")
                    except Exception as e:
                        logger.error(f"Error removing expired file: {e}")
                        
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            'file_cache_enabled': True,
            'redis_cache_enabled': self.redis_available,
            'model_cache_count': len(self.model_cache),
            'max_model_cache_size': self.model_cache_size
        }
        
        # File cache stats
        try:
            total_size = 0
            file_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(filepath)
                    file_count += 1
            
            stats.update({
                'file_cache_count': file_count,
                'file_cache_size_mb': round(total_size / (1024 * 1024), 2)
            })
        except Exception as e:
            logger.error(f"Error getting file cache stats: {e}")
        
        # Redis stats
        if self.redis_available:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used_mb': round(info.get('used_memory', 0) / (1024 * 1024), 2),
                    'redis_connected_clients': info.get('connected_clients', 0)
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        
        return stats
    
    def clear_cache(self, cache_type: str = 'all'):
        """Clear cache data."""
        if cache_type in ['all', 'files']:
            # Clear file cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, filename))
                logger.info("File cache cleared")
            except Exception as e:
                logger.error(f"Error clearing file cache: {e}")
        
        if cache_type in ['all', 'models']:
            # Clear model cache
            self.model_cache.clear()
            logger.info("Model cache cleared")
        
        if cache_type in ['all', 'redis'] and self.redis_available:
            # Clear Redis cache
            try:
                self.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")