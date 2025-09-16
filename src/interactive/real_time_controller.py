"""
Real-time Style Transfer Controller
==================================

Real-time audio processing and style transfer control system.
"""

import logging
import threading
import queue
import time
from typing import Dict, Any, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

class RealTimeController:
    """Real-time controller for style transfer operations."""
    
    def __init__(self, buffer_size: int = 1024, sample_rate: int = 22050):
        """
        Initialize real-time controller.
        
        Args:
            buffer_size: Audio buffer size for real-time processing
            sample_rate: Audio sample rate
        """
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.callbacks = {}
        
        logger.info(f"RealTimeController initialized with buffer_size={buffer_size}, sr={sample_rate}")
    
    def start_processing(self, style_transfer_model=None):
        """Start real-time processing thread."""
        if self.is_running:
            logger.warning("Processing already running")
            return
        
        self.is_running = True
        self.style_transfer_model = style_transfer_model
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Real-time processing stopped")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float = None):
        """
        Add audio chunk for processing.
        
        Args:
            audio_chunk: Audio data chunk
            timestamp: Optional timestamp for the chunk
        """
        if not self.is_running:
            logger.warning("Controller not running, ignoring audio chunk")
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            self.audio_queue.put({
                'audio': audio_chunk,
                'timestamp': timestamp,
                'chunk_id': f"chunk_{int(timestamp * 1000)}"
            }, timeout=0.1)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")
    
    def get_processed_chunk(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get processed audio chunk.
        
        Args:
            timeout: Timeout for queue get operation
            
        Returns:
            Processed audio chunk data or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get audio chunk from queue
                chunk_data = self.audio_queue.get(timeout=0.1)
                
                # Process the chunk
                processed_chunk = self._process_chunk(chunk_data)
                
                # Put result in output queue
                try:
                    self.result_queue.put(processed_chunk, timeout=0.1)
                except queue.Full:
                    logger.warning("Result queue full, dropping processed chunk")
                
                # Call callbacks if registered
                self._call_callbacks('chunk_processed', processed_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue
        
        logger.info("Processing loop ended")
    
    def _process_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single audio chunk.
        
        Args:
            chunk_data: Input chunk data
            
        Returns:
            Processed chunk data
        """
        try:
            audio = chunk_data['audio']
            timestamp = chunk_data['timestamp']
            chunk_id = chunk_data['chunk_id']
            
            # Basic processing (placeholder)
            if hasattr(self, 'style_transfer_model') and self.style_transfer_model:
                # Use the style transfer model if available
                # This is a simplified version for real-time processing
                processed_audio = self._apply_real_time_style_transfer(audio)
            else:
                # Simple pass-through with basic effects
                processed_audio = self._apply_basic_effects(audio)
            
            return {
                'audio': processed_audio,
                'original_audio': audio,
                'timestamp': timestamp,
                'chunk_id': chunk_id,
                'processing_time': time.time() - timestamp,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return {
                'audio': chunk_data['audio'],  # Return original on error
                'original_audio': chunk_data['audio'],
                'timestamp': chunk_data['timestamp'],
                'chunk_id': chunk_data['chunk_id'],
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def _apply_real_time_style_transfer(self, audio: np.ndarray) -> np.ndarray:
        """Apply style transfer for real-time processing."""
        # This is a simplified version optimized for real-time processing
        # In a full implementation, you would use a lightweight version
        # of your style transfer model
        
        # Basic style effects for demonstration
        # Apply some distortion for "rock" style
        gain = 1.2
        processed = np.tanh(audio * gain) * 0.8
        
        # Add some harmonics
        if len(audio) > 0:
            harmonics = np.sin(2 * np.pi * np.arange(len(audio)) / self.sample_rate * 220) * 0.1
            processed = processed + harmonics[:len(processed)]
        
        # Normalize
        if np.max(np.abs(processed)) > 0:
            processed = processed / np.max(np.abs(processed)) * 0.9
        
        return processed
    
    def _apply_basic_effects(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic audio effects."""
        # Simple gain and normalization
        processed = audio * 1.1
        
        # Basic compression simulation
        threshold = 0.7
        compressed = np.where(
            np.abs(processed) > threshold,
            np.sign(processed) * (threshold + (np.abs(processed) - threshold) * 0.3),
            processed
        )
        
        # Normalize
        if np.max(np.abs(compressed)) > 0:
            compressed = compressed / np.max(np.abs(compressed)) * 0.9
        
        return compressed
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for specific events.
        
        Args:
            event: Event name ('chunk_processed', 'error', etc.)
            callback: Callback function
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
        logger.info(f"Registered callback for event: {event}")
    
    def _call_callbacks(self, event: str, data: Any):
        """Call all registered callbacks for an event."""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error for event {event}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            'is_running': self.is_running,
            'audio_queue_size': self.audio_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'buffer_size': self.buffer_size,
            'sample_rate': self.sample_rate,
            'has_model': hasattr(self, 'style_transfer_model') and self.style_transfer_model is not None
        }
    
    def set_parameters(self, **params):
        """Set processing parameters."""
        if 'buffer_size' in params:
            self.buffer_size = params['buffer_size']
            logger.info(f"Buffer size updated to: {self.buffer_size}")
        
        if 'sample_rate' in params:
            self.sample_rate = params['sample_rate']
            logger.info(f"Sample rate updated to: {self.sample_rate}")
    
    def clear_queues(self):
        """Clear all processing queues."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Queues cleared")