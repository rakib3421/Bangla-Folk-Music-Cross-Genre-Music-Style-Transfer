"""
Main Application Runner
======================

Entry point for the Flask application with proper initialization.
"""

import os
import sys
import logging
from app import create_app
from app.config import config

def setup_logging(app):
    """Configure logging for the application."""
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)

def main():
    """Main entry point."""
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Create app and socketio
    app, socketio = create_app(config_name)
    
    # Setup logging
    setup_logging(app)
    
    # Get host and port from environment or command line
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    debug = app.config.get('DEBUG', False)
    
    # Print startup information
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  Bangla Folk Style Transfer                  ║
║                     Web Application                          ║
╠══════════════════════════════════════════════════════════════╣
║ Environment: {config_name.ljust(45)} ║
║ Host:        {host.ljust(45)} ║
║ Port:        {str(port).ljust(45)} ║
║ Debug:       {str(debug).ljust(45)} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🎵 Starting Style Transfer Server...")
    print(f"🌐 Access the web interface at: http://{host}:{port}")
    print(f"📡 API documentation at: http://{host}:{port}/api/v1/info")
    print(f"🔥 Ready to transform your music!\n")
    
    try:
        # Start the application
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug,
            log_output=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()