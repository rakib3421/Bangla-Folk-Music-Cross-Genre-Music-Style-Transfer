"""
Authentication Manager for Style Transfer API
===========================================

Handles user authentication, API key management, and session control.
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import os

class AuthManager:
    """Manages user authentication and API keys."""
    
    def __init__(self, users_file='users.json', api_keys_file='api_keys.json'):
        """Initialize authentication manager."""
        self.users_file = users_file
        self.api_keys_file = api_keys_file
        self.users = self._load_users()
        self.api_keys = self._load_api_keys()
        self.sessions = {}  # Active sessions
    
    def _load_users(self) -> Dict[str, Any]:
        """Load users from file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_users(self):
        """Save users to file."""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _load_api_keys(self) -> Dict[str, Any]:
        """Load API keys from file."""
        if os.path.exists(self.api_keys_file):
            try:
                with open(self.api_keys_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_api_keys(self):
        """Save API keys to file."""
        with open(self.api_keys_file, 'w') as f:
            json.dump(self.api_keys, f, indent=2)
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode(), 
                                          salt.encode(), 
                                          100000)
        return password_hash.hex(), salt
    
    def register_user(self, username: str, password: str, email: str = None) -> bool:
        """Register a new user."""
        if username in self.users:
            return False
        
        password_hash, salt = self._hash_password(password)
        
        self.users[username] = {
            'password_hash': password_hash,
            'salt': salt,
            'email': email,
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            'api_quota': 100,  # Default API calls per day
            'api_calls_today': 0,
            'last_api_reset': datetime.now().date().isoformat()
        }
        
        self._save_users()
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user with username/password."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        if not user.get('is_active', True):
            return False
        
        password_hash, _ = self._hash_password(password, user['salt'])
        return password_hash == user['password_hash']
    
    def create_session(self, username: str) -> str:
        """Create a new session for authenticated user."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'username': username,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return username if valid."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if datetime.now() > session['expires_at']:
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return session['username']
    
    def logout(self, session_id: str) -> bool:
        """Logout user by removing session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def generate_api_key(self, username: str, description: str = None) -> str:
        """Generate API key for user."""
        if username not in self.users:
            raise ValueError("User not found")
        
        api_key = f"st_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            'username': username,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            'usage_count': 0,
            'last_used': None
        }
        
        self._save_api_keys()
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return username if valid."""
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        
        if not key_info.get('is_active', True):
            return None
        
        username = key_info['username']
        
        # Check if user is active
        if username not in self.users or not self.users[username].get('is_active', True):
            return None
        
        # Update usage
        key_info['usage_count'] += 1
        key_info['last_used'] = datetime.now().isoformat()
        self._save_api_keys()
        
        return username
    
    def check_api_quota(self, username: str) -> bool:
        """Check if user has remaining API quota."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        today = datetime.now().date().isoformat()
        
        # Reset daily counter if it's a new day
        if user.get('last_api_reset') != today:
            user['api_calls_today'] = 0
            user['last_api_reset'] = today
            self._save_users()
        
        return user['api_calls_today'] < user.get('api_quota', 100)
    
    def increment_api_usage(self, username: str):
        """Increment API usage for user."""
        if username in self.users:
            self.users[username]['api_calls_today'] += 1
            self._save_users()
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]['is_active'] = False
            self._save_api_keys()
            return True
        return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)."""
        if username not in self.users:
            return None
        
        user = self.users[username].copy()
        # Remove sensitive information
        user.pop('password_hash', None)
        user.pop('salt', None)
        
        return user
    
    def get_user_api_keys(self, username: str) -> list:
        """Get all API keys for a user."""
        return [
            {
                'key': key[:8] + '...',  # Show only first 8 characters
                'description': info.get('description'),
                'created_at': info.get('created_at'),
                'usage_count': info.get('usage_count', 0),
                'last_used': info.get('last_used'),
                'is_active': info.get('is_active', True)
            }
            for key, info in self.api_keys.items()
            if info.get('username') == username
        ]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if now > session['expires_at']
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]