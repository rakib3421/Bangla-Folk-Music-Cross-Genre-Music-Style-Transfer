#!/bin/bash

# Deployment script for Style Transfer application
# This script sets up the production environment

set -e

echo "🚀 Starting Style Transfer Application Deployment"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads cache logs ssl

# Set permissions
chmod 755 uploads cache logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before running the application"
fi

# Generate secret key if not set
if ! grep -q "SECRET_KEY=your-secret-key-here" .env; then
    echo "🔑 Generating secret key..."
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s/SECRET_KEY=your-secret-key-here-change-in-production/SECRET_KEY=$SECRET_KEY/" .env
fi

# Build and start services
echo "🏗️  Building application..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🎵 Style Transfer Application is running!"
    echo "🌐 Web Interface: http://localhost"
    echo "📡 API Endpoint: http://localhost/api/v1"
    echo "📊 Health Check: http://localhost/api/v1/health"
    echo ""
    echo "📝 View logs with: docker-compose logs -f"
    echo "🛑 Stop with: docker-compose down"
else
    echo "❌ Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi