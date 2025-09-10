# logging_config.py - Enhanced logging with UTF-8 support and emoji handling

import logging
import sys
import io
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=True, log_file="trading.log"):
    """Setup comprehensive logging with UTF-8 support for Windows"""
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console Handler with UTF-8 support
    try:
        # For Windows, wrap stdout with UTF-8 encoding
        if sys.platform.startswith('win'):
            try:
                utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                console_handler = logging.StreamHandler(utf8_stdout)
            except (AttributeError, io.UnsupportedOperation):
                console_handler = logging.StreamHandler(sys.stdout)
        else:
            # For Unix/Linux systems
            console_handler = logging.StreamHandler(sys.stdout)
        
        # Add emoji filter for console output
        class EmojiFilter(logging.Filter):
            def filter(self, record):
                if hasattr(record, 'msg'):
                    # Replace common emojis with ASCII equivalents for better compatibility
                    emoji_replacements = {
                        '🚀': '[START]',
                        '✅': '[OK]',
                        '🔌': '[CONN]',
                        '⏳': '[WAIT]',
                        '📡': '[WS]',
                        '🎯': '[TARGET]',
                        '📊': '[DATA]',
                        '⚡': '[FAST]',
                        '🔄': '[SYNC]',
                        '❌': '[ERROR]',
                        '⚠️': '[WARN]',
                        '💰': '[MONEY]',
                        '🔍': '[SEARCH]',
                        '🏆': '[WIN]',
                        '🛑': '[STOP]',
                        '🔧': '[SETUP]',
                        '🔥': '[HOT]',
                        '📈': '[UP]',
                        '📉': '[DOWN]',
                        '🧹': '[CLEAN]'
                    }
                    
                    msg = str(record.msg)
                    for emoji, replacement in emoji_replacements.items():
                        msg = msg.replace(emoji, replacement)
                    record.msg = msg
                return True
        
        console_handler.addFilter(EmojiFilter())
        console_handler.setLevel(log_level)
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
    except Exception as e:
        print(f"Warning: Could not setup console logging: {e}")
    
    # File Handler (UTF-8 by default)
    if log_to_file:
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs'
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Use timestamp in filename if requested
            if log_file == "trading.log":
                timestamp = datetime.now().strftime("%Y%m%d")
                log_file = f"logs/trading_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # File gets more detailed logs
            
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            print(f"✅ File logging enabled: {log_file}")
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)

def test_unicode_support():
    """Test if the current console supports Unicode"""
    try:
        test_message = "🚀 Unicode test - if you see a rocket, Unicode is working! ✅"
        print(test_message)
        return True
    except UnicodeEncodeError:
        print("[START] Unicode test - emojis will be replaced with ASCII [OK]")
        return False

def setup_module_logging():
    """Setup specific logging levels for different modules"""
    # Trading system modules
    logging.getLogger('orchestrator').setLevel(logging.INFO)
    logging.getLogger('fractal_order_flow_strategy').setLevel(logging.INFO)
    logging.getLogger('data_adapter').setLevel(logging.INFO)
    logging.getLogger('websocket_client').setLevel(logging.WARNING)
    logging.getLogger('pnl_tracker').setLevel(logging.INFO)
    logging.getLogger('supervisor').setLevel(logging.INFO)
    
    # External libraries
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('requests.packages.urllib3').setLevel(logging.WARNING)

if __name__ == "__main__":
    setup_logging()
    setup_module_logging()
    test_unicode_support()
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("🚀 Logging system initialized successfully!")
    logger.debug("Debug message test")
    logger.warning("⚠️ Warning message test")
    logger.error("❌ Error message test")
