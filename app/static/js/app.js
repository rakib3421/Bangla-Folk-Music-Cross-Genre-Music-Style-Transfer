/**
 * Bangla Folk Style Transfer - Frontend Application
 * Handles file upload, style selection, real-time updates, and user interface
 */

class StyleTransferApp {
    constructor() {
        this.socket = null;
        this.currentTask = null;
        this.selectedStyle = 'rock';
        this.intensity = 0.7;
        this.uploadedFile = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadHistory();
        this.showSection('upload');
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = link.getAttribute('href').substring(1);
                this.showSection(target);
                this.updateNavigation(link);
            });
        });
        
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('audioFile');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        
        // Remove file
        document.getElementById('removeFile').addEventListener('click', () => {
            this.removeFile();
        });
        
        // Style selection
        document.querySelectorAll('.style-option').forEach(option => {
            option.addEventListener('click', () => {
                this.selectStyle(option.dataset.style);
                this.updateStyleSelection(option);
            });
        });
        
        // Intensity slider
        const intensitySlider = document.getElementById('intensitySlider');
        intensitySlider.addEventListener('input', (e) => {
            this.intensity = parseFloat(e.target.value);
            document.getElementById('intensityValue').textContent = 
                Math.round(this.intensity * 100) + '%';
        });
        
        // Transform button
        document.getElementById('transformBtn').addEventListener('click', () => {
            this.startTransform();
        });
        
        // Cancel processing
        document.getElementById('cancelBtn').addEventListener('click', () => {
            this.cancelProcessing();
        });
        
        // Download result
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadResult();
        });
        
        // New transform
        document.getElementById('newTransformBtn').addEventListener('click', () => {
            this.resetForNewTransform();
        });
        
        // API key generation
        document.getElementById('generateKeyBtn').addEventListener('click', () => {
            this.generateApiKey();
        });
        
        // Modal close
        document.getElementById('closeErrorModal').addEventListener('click', () => {
            this.hideModal('errorModal');
        });
        document.getElementById('closeErrorBtn').addEventListener('click', () => {
            this.hideModal('errorModal');
        });
    }
    
    setupWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });
        
        this.socket.on('task_update', (data) => {
            this.handleTaskUpdate(data);
        });
        
        this.socket.on('status', (data) => {
            console.log('Status:', data.message);
        });
    }
    
    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
        }
    }
    
    updateNavigation(activeLink) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        activeLink.classList.add('active');
    }
    
    handleFileSelect(file) {
        // Validate file type
        const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/mp4', 'audio/ogg'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid audio file (MP3, WAV, FLAC, M4A, OGG)');
            return;
        }
        
        // Validate file size (50MB max)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size must be less than 50MB');
            return;
        }
        
        this.uploadedFile = file;
        
        // Show file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
        
        // Create audio preview
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = URL.createObjectURL(file);
        
        // Show file info and hide upload area
        document.getElementById('fileInfo').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
        
        // Enable transform button if style is selected
        this.updateTransformButton();
    }
    
    removeFile() {
        this.uploadedFile = null;
        
        // Hide file info and show upload area
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('uploadArea').style.display = 'block';
        
        // Clear audio player
        document.getElementById('audioPlayer').src = '';
        
        // Disable transform button
        this.updateTransformButton();
    }
    
    selectStyle(style) {
        this.selectedStyle = style;
        this.updateTransformButton();
    }
    
    updateStyleSelection(selectedOption) {
        document.querySelectorAll('.style-option').forEach(option => {
            option.classList.remove('selected');
        });
        selectedOption.classList.add('selected');
    }
    
    updateTransformButton() {
        const transformBtn = document.getElementById('transformBtn');
        const canTransform = this.uploadedFile && this.selectedStyle;
        
        transformBtn.disabled = !canTransform;
    }
    
    async startTransform() {
        if (!this.uploadedFile || !this.selectedStyle) {
            this.showError('Please select an audio file and target style');
            return;
        }
        
        // Show loading
        this.showLoading();
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('audio_file', this.uploadedFile);
            formData.append('target_style', this.selectedStyle);
            formData.append('intensity', this.intensity.toString());
            
            // Send upload request
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.currentTask = result.task_id;
                
                if (result.status === 'completed' && result.cached) {
                    // Show result immediately if cached
                    this.showResults(result.result_url);
                } else {
                    // Subscribe to task updates
                    this.socket.emit('subscribe_task', { task_id: this.currentTask });
                    
                    // Show processing section
                    this.showSection('processing');
                    this.updateProgress(0, 'Starting processing...');
                }
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }
    
    handleTaskUpdate(data) {
        if (data.task_id === this.currentTask) {
            if (data.status === 'processing') {
                this.updateProgress(data.progress, data.message);
            } else if (data.status === 'completed') {
                this.showResults(data.download_url);
            } else if (data.status === 'failed') {
                this.showError(data.message);
                this.showSection('upload');
            }
        }
    }
    
    updateProgress(progress, message) {
        document.getElementById('progressFill').style.width = progress + '%';
        document.getElementById('progressPercent').textContent = progress + '%';
        document.getElementById('processingMessage').textContent = message;
    }
    
    showResults(downloadUrl) {
        // Set up audio players
        const originalAudio = document.getElementById('originalAudio');
        const transformedAudio = document.getElementById('transformedAudio');
        
        originalAudio.src = URL.createObjectURL(this.uploadedFile);
        
        // Set transformed audio source - extract task ID from download URL
        const taskId = downloadUrl.split('/').pop();
        transformedAudio.src = `/audio/${taskId}`;
        
        // Set download URL
        document.getElementById('downloadBtn').onclick = () => {
            window.open(downloadUrl, '_blank');
        };
        
        // Show results section
        this.showSection('results');
        
        // Add to history
        this.addToHistory();
    }
    
    cancelProcessing() {
        if (this.currentTask) {
            // Here you would implement task cancellation
            // For now, just return to upload
            this.currentTask = null;
            this.showSection('upload');
        }
    }
    
    downloadResult() {
        if (this.currentTask) {
            window.open(`/download/${this.currentTask}`, '_blank');
        }
    }
    
    resetForNewTransform() {
        this.currentTask = null;
        this.removeFile();
        this.showSection('upload');
        
        // Reset form
        document.querySelectorAll('.style-option').forEach(option => {
            option.classList.remove('selected');
        });
        
        // Select first style by default
        const firstStyle = document.querySelector('.style-option[data-style="rock"]');
        if (firstStyle) {
            firstStyle.click();
        }
        
        // Reset intensity
        document.getElementById('intensitySlider').value = '0.7';
        document.getElementById('intensityValue').textContent = '70%';
        this.intensity = 0.7;
    }
    
    addToHistory() {
        const historyItem = {
            id: this.currentTask,
            filename: this.uploadedFile.name,
            style: this.selectedStyle,
            intensity: this.intensity,
            date: new Date().toLocaleDateString(),
            time: new Date().toLocaleTimeString()
        };
        
        // Get existing history
        let history = JSON.parse(localStorage.getItem('styleTransferHistory') || '[]');
        
        // Add new item to beginning
        history.unshift(historyItem);
        
        // Keep only last 20 items
        history = history.slice(0, 20);
        
        // Save to localStorage
        localStorage.setItem('styleTransferHistory', JSON.stringify(history));
        
        // Update history display
        this.loadHistory();
    }
    
    loadHistory() {
        const historyList = document.getElementById('historyList');
        const history = JSON.parse(localStorage.getItem('styleTransferHistory') || '[]');
        
        if (history.length === 0) {
            historyList.innerHTML = '<p style="text-align: center; color: #666;">No transformations yet</p>';
            return;
        }
        
        historyList.innerHTML = history.map(item => `
            <div class="history-item">
                <div class="history-header">
                    <span class="history-title">${item.filename}</span>
                    <span class="history-date">${item.date} ${item.time}</span>
                </div>
                <div class="history-details">
                    <div class="history-detail">
                        <strong>Style</strong>
                        <span>${this.capitalizeFirst(item.style)}</span>
                    </div>
                    <div class="history-detail">
                        <strong>Intensity</strong>
                        <span>${Math.round(item.intensity * 100)}%</span>
                    </div>
                    <div class="history-detail">
                        <strong>Status</strong>
                        <span>Completed</span>
                    </div>
                </div>
                <div class="history-actions">
                    <button class="btn-small btn-primary" onclick="app.downloadHistoryItem('${item.id}')">
                        Download
                    </button>
                    <button class="btn-small btn-secondary" onclick="app.removeHistoryItem('${item.id}')">
                        Remove
                    </button>
                </div>
            </div>
        `).join('');
    }
    
    downloadHistoryItem(taskId) {
        window.open(`/download/${taskId}`, '_blank');
    }
    
    removeHistoryItem(taskId) {
        let history = JSON.parse(localStorage.getItem('styleTransferHistory') || '[]');
        history = history.filter(item => item.id !== taskId);
        localStorage.setItem('styleTransferHistory', JSON.stringify(history));
        this.loadHistory();
    }
    
    async generateApiKey() {
        try {
            this.showLoading();
            
            const response = await fetch('/api/v1/auth/generate-key', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    description: 'Web Interface Generated Key'
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showApiKey(result.api_key);
            } else {
                throw new Error(result.error || 'Failed to generate API key');
            }
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }
    
    showApiKey(apiKey) {
        const apiKeysContainer = document.getElementById('apiKeys');
        const keyElement = document.createElement('div');
        keyElement.className = 'api-key-item';
        keyElement.innerHTML = `
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <h4>New API Key Generated</h4>
                <p style="font-family: monospace; background: white; padding: 10px; border-radius: 4px; word-break: break-all;">
                    ${apiKey}
                </p>
                <p style="color: #dc3545; font-size: 0.9rem; margin-top: 10px;">
                    <strong>Important:</strong> Copy this key now. You won't be able to see it again.
                </p>
            </div>
        `;
        apiKeysContainer.appendChild(keyElement);
    }
    
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.showModal('errorModal');
    }
    
    showModal(modalId) {
        document.getElementById(modalId).classList.add('show');
    }
    
    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('show');
    }
    
    showLoading() {
        document.getElementById('loadingOverlay').classList.add('show');
    }
    
    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('show');
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StyleTransferApp();
});

// Service Worker for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}