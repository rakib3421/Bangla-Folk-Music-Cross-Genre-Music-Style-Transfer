# Bangla Folk Style Transfer - Web Application

A comprehensive web application for transforming Bangla folk music into Rock and Jazz styles using advanced neural audio processing.

## 🎵 Features

### Web Interface
- **Drag & Drop File Upload**: Easy audio file upload with support for MP3, WAV, FLAC, M4A, OGG
- **Style Selection**: Choose between Rock, Jazz, or blended transformation styles
- **Real-time Processing**: Live progress updates and status tracking
- **Interactive Preview**: Compare original and transformed audio side-by-side
- **Processing History**: Track and manage your transformations

### RESTful API
- **Batch Processing**: Upload and process multiple files simultaneously
- **Authentication**: Secure API key-based authentication system
- **Rate Limiting**: Built-in protection against abuse
- **Status Tracking**: Real-time status updates for processing tasks
- **WebSocket Support**: Live updates for web and API clients

### Performance Features
- **Model Caching**: Intelligent caching for faster inference
- **Result Caching**: Cache processed results for repeated requests
- **Progressive Loading**: Efficient handling of large audio files
- **Redis Integration**: Fast caching and session management

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Prerequisites**
   ```bash
   # Install Docker and Docker Compose
   # https://docs.docker.com/get-docker/
   ```

2. **Deploy with Docker**
   ```bash
   # Linux/macOS
   chmod +x deploy.sh
   ./deploy.sh
   
   # Windows
   deploy.bat
   ```

3. **Access the Application**
   - Web Interface: http://localhost
   - API Documentation: http://localhost/api/v1/info
   - Health Check: http://localhost/api/v1/health

### Option 2: Local Development

1. **Prerequisites**
   ```bash
   # Python 3.11+
   python --version
   
   # Redis (optional but recommended)
   # Install from: https://redis.io/download
   ```

2. **Setup Environment**
   ```bash
   # Clone and navigate to project
   git clone <repository-url>
   cd Project-3
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-web.txt
   ```

3. **Configure Environment**
   ```bash
   # Copy and edit environment file
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run Application**
   ```bash
   python run_app.py
   ```

## 📡 API Usage

### Authentication

1. **Register User**
   ```bash
   curl -X POST http://localhost:5000/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{"username": "your_username", "password": "your_password", "email": "your@email.com"}'
   ```

2. **Generate API Key**
   ```bash
   curl -X POST http://localhost:5000/api/v1/auth/generate-key \
     -H "Content-Type: application/json" \
     -u "your_username:your_password" \
     -d '{"description": "My API Key"}'
   ```

### File Upload and Processing

1. **Upload Audio File**
   ```bash
   curl -X POST http://localhost:5000/api/v1/upload \
     -H "X-API-Key: your_api_key" \
     -F "audio_file=@path/to/your/audio.mp3" \
     -F "target_style=rock" \
     -F "intensity=0.7"
   ```

2. **Check Processing Status**
   ```bash
   curl -X GET http://localhost:5000/api/v1/status/TASK_ID \
     -H "X-API-Key: your_api_key"
   ```

3. **Download Result**
   ```bash
   curl -X GET http://localhost:5000/api/v1/download/TASK_ID \
     -H "X-API-Key: your_api_key" \
     -o transformed_audio.wav
   ```

### Batch Processing

```bash
curl -X POST http://localhost:5000/api/v1/batch \
  -H "X-API-Key: your_api_key" \
  -F "audio_files=@file1.mp3" \
  -F "audio_files=@file2.mp3" \
  -F "target_style=jazz" \
  -F "intensity=0.8"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment (development/production) | development |
| `SECRET_KEY` | Flask secret key | auto-generated |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379/0 |
| `MAX_CONTENT_LENGTH` | Maximum file size (bytes) | 52428800 (50MB) |
| `CACHE_TTL_HOURS` | Cache expiration time | 24 |
| `DEFAULT_RATE_LIMIT` | API rate limit | 200 per day, 50 per hour |

### Supported Audio Formats

- **MP3**: Most common format, good compression
- **WAV**: Uncompressed, best quality
- **FLAC**: Lossless compression
- **M4A**: Apple's format
- **OGG**: Open-source alternative

### Style Options

- **Rock**: Powerful drums, electric guitars, energetic rhythm
- **Jazz**: Smooth improvisation, complex harmonies, swing rhythm  
- **Blend**: Fusion of Rock and Jazz characteristics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   Mobile App    │    │  API Client     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Nginx (Reverse Proxy)             │
         └─────────────────────┬───────────────────────────┘
                               │
         ┌─────────────────────────────────────────────────┐
         │             Flask Application                   │
         │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
         │  │ Web Routes  │  │ API Routes  │  │ WebSocket│ │
         │  └─────────────┘  └─────────────┘  └──────────┘ │
         └─────────────────────┬───────────────────────────┘
                               │
    ┌────────────┬─────────────┼─────────────┬──────────────┐
    │            │             │             │              │
┌───▼───┐  ┌────▼────┐  ┌─────▼─────┐  ┌───▼───┐  ┌──────▼──────┐
│ Redis │  │ Model   │  │ Processing│  │ Cache │  │ Style       │
│ Cache │  │ Cache   │  │ Queue     │  │ Mgr   │  │ Transfer    │
└───────┘  └─────────┘  └───────────┘  └───────┘  │ Engine      │
                                                  └─────────────┘
```

## 🔒 Security Features

- **API Key Authentication**: Secure access control
- **Rate Limiting**: Protection against abuse
- **File Validation**: Strict file type and size checking
- **CORS Protection**: Configurable cross-origin policies
- **Security Headers**: XSS, CSRF, and other protections
- **Session Management**: Secure session handling

## 📊 Monitoring and Logging

### Health Checks
```bash
# Application health
curl http://localhost:5000/api/v1/health

# Cache statistics
curl -H "X-API-Key: your_key" http://localhost:5000/api/v1/cache/stats
```

### Log Files
- **Application Logs**: `logs/app.log`
- **Error Logs**: `logs/error.log`
- **Access Logs**: Available through Nginx in production

## 🐳 Docker Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Copy and configure environment
   cp .env.example .env
   # Edit production settings in .env
   ```

2. **SSL Configuration** (Optional)
   ```bash
   # Place SSL certificates in ssl/ directory
   ssl/cert.pem
   ssl/key.pem
   ```

3. **Deploy**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

### Scaling

```bash
# Scale application instances
docker-compose up -d --scale app=3

# Scale with load balancer
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

## 🛠️ Development

### Local Development Setup

1. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Code Formatting**
   ```bash
   black app/
   flake8 app/
   mypy app/
   ```

4. **Development Server with Auto-reload**
   ```bash
   FLASK_ENV=development python run_app.py
   ```

### API Testing

Use the included Postman collection or test with curl:

```bash
# Run integration tests
python -m pytest tests/test_api.py

# Load testing
python tests/load_test.py
```

## 📈 Performance Optimization

### Caching Strategy
- **Model Caching**: Keep 3 models in memory
- **Result Caching**: 24-hour TTL for processed files
- **Redis Caching**: Fast access to metadata and sessions

### Resource Management
- **Memory Limits**: Configure based on available RAM
- **Processing Queue**: Limit concurrent processing
- **File Cleanup**: Automatic cleanup of old files

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Start Redis
   redis-server
   ```

2. **File Upload Fails**
   - Check file size (max 50MB)
   - Verify file format is supported
   - Check disk space in uploads directory

3. **Processing Timeout**
   - Increase `PROCESSING_TIMEOUT` in environment
   - Check system resources (CPU, Memory)
   - Verify audio file is not corrupted

4. **API Rate Limiting**
   - Check your API usage in profile
   - Upgrade your plan if needed
   - Implement exponential backoff in clients

### Log Analysis

```bash
# View recent logs
docker-compose logs -f app

# Search for errors
grep ERROR logs/app.log

# Monitor processing queue
curl -H "X-API-Key: your_key" http://localhost:5000/api/v1/cache/stats
```

## 📞 Support

- **Documentation**: [API Documentation](http://localhost:5000/api/v1/info)
- **Issues**: Report bugs and feature requests in the repository
- **Community**: Join our Discord/Slack for community support

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ for music lovers and cultural preservation**