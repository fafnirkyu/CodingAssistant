import time
import json
import inspect
import logging
import traceback
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler

# Ultra-Robust Debug Logger with Multiple Fallbacks

class UltraRobustDebugLogger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._setup_logging()
    
    def _safe_path_creation(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"⚠️ Cannot create log directory {path.parent}: {e}")
            return False
    
    def _setup_logging(self):
        self.log_file = self._get_safe_log_path()
        try:
            # Clear any existing configuration
            logging.root.handlers.clear()
            
            # Create logger
            self.logger = logging.getLogger('ultra_robust_debug')
            self.logger.setLevel(logging.DEBUG)
            self.logger.handlers.clear()  # Clear existing handlers
            self.logger.propagate = False  # Prevent double logging
            
            # Create formatter with thread name
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)-8s] [%(threadName)s] %(name)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler (always available)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (rotating) with fallback
            if self.log_file and self._safe_path_creation(self.log_file):
                try:
                    file_handler = RotatingFileHandler(
                        filename=str(self.log_file),
                        maxBytes=5_000_000,
                        backupCount=3,
                        encoding='utf-8',
                        mode='a'
                    )
                    file_handler.setLevel(logging.DEBUG)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                    self._log_success("File logging enabled (rotating)")
                except Exception as e:
                    self._log_warning(f"File logging disabled: {e}")
            
            self._log_success("Debug logger initialized successfully")
        except Exception as e:
            # Ultimate fallback - basic print logging
            print(f"CRITICAL: Debug logger setup failed: {e}")
            self.logger = None
    
    def _get_safe_log_path(self) -> Optional[Path]:
        possible_paths = [
            Path("./logs/debug.log"),
        ]
        for path in possible_paths:
            try:
                if self._safe_path_creation(path):
                    return path
            except Exception:
                continue
        return None
    
    def _log_success(self, message: str):
        if self.logger:
            try:
                self.logger.info(f"✅ {message}")
            except Exception:
                pass
        print(f"✅ {message}")
    
    def _log_warning(self, message: str):
        if self.logger:
            try:
                self.logger.warning(f"⚠️ {message}")
            except Exception:
                pass
        print(f"⚠️ {message}")
    
    def _log_error(self, message: str):
        if self.logger:
            try:
                self.logger.error(f"{message}")
            except Exception:
                pass
        print(f"{message}")
    
    def _safe_serialize(self, obj: Any, max_length: int = 400) -> str:
        try:
            if obj is None:
                return "None"
            elif isinstance(obj, (str, int, float, bool)):
                s = str(obj)
                return s if len(s) <= max_length else s[:max_length] + "..."
            elif isinstance(obj, (dict, list, tuple)):
                serialized = json.dumps(obj, ensure_ascii=False, default=str)
                return serialized if len(serialized) <= max_length else serialized[:max_length] + "..."
            else:
                s = str(obj)
                return s if len(s) <= max_length else s[:max_length] + "..."
        except Exception as e:
            return f"[SerializationError: {e}]"
    
    def log_debug(self, message: str, extra_data: Optional[Dict] = None):
        try:
            full_message = message
            if extra_data:
                extra_str = self._safe_serialize(extra_data)
                full_message = f"{message} | extra={extra_str}"
            
            if self.logger:
                try:
                    self.logger.debug(full_message)
                except Exception:
                    pass
            print(f"🔍 {full_message}")
        except Exception as e:
            print(f"LOGGING FAILED: {e} | Original message: {message}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        try:
            full_message = message
            if exception:
                try:
                    tb_list = traceback.format_exception(type(exception), exception, exception.__traceback__)
                    tb_str = "".join(tb_list)
                except Exception:
                    tb_str = traceback.format_exc()
                full_message = f"{message} | exception={repr(exception)} | traceback={tb_str}"
            
            if self.logger:
                try:
                    self.logger.error(full_message)
                except Exception:
                    pass
            print(f"{full_message}")
        except Exception as e:
            print(f"ERROR LOGGING FAILED: {e} | Original: {message}")
    
    def trace_function(self, log_args: bool = True, log_result: bool = True, log_time: bool = True):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__qualname__
                module_name = func.__module__
                full_name = f"{module_name}.{func_name}" if module_name != "__main__" else func_name
                
                # Log function entry
                entry_data = {}
                if log_args:
                    try:
                        arg_spec = inspect.signature(func)
                        arg_names = list(arg_spec.parameters.keys())
                        arg_data = {}
                        
                        # Handle positional arguments
                        for i, (name, value) in enumerate(zip(arg_names, args)):
                            arg_data[name] = self._safe_serialize(value)
                        
                        # Handle keyword arguments
                        for name, value in kwargs.items():
                            arg_data[name] = self._safe_serialize(value)
                        
                        entry_data["args"] = arg_data
                    except Exception as e:
                        entry_data["args_error"] = f"Failed to parse: {e}"
                
                self.log_debug(f"➡️ ENTER {full_name}", entry_data)
                
                # Execute function with timing
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time if log_time else 0
                    
                    # Log function exit
                    exit_data = {}
                    if log_time:
                        exit_data["duration_seconds"] = f"{elapsed:.3f}"
                    
                    if log_result and result is not None:
                        exit_data["result"] = self._safe_serialize(result)
                    
                    status_icon = "✅" if not isinstance(result, Exception) else "⚠️"
                    self.log_debug(f"{status_icon} EXIT {full_name}", exit_data)
                    
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time if log_time else 0
                    # Log exception with traceback
                    error_data = {"duration_seconds": f"{elapsed:.3f}"} if log_time else {}
                    self.log_error(f"💥 EXCEPTION in {full_name}", e)
                    raise
            return wrapper
        return decorator
    
    # Utility methods for specific log types
    def log_http_request(self, method: str, url: str, headers: Optional[Dict] = None, body: Optional[Any] = None):
        extra = {
            "http_method": method,
            "http_url": url,
            "http_headers": self._safe_serialize(headers) if headers else None,
            "http_body": self._safe_serialize(body) if body else None
        }
        self.log_debug("🌐 HTTP REQUEST", extra)
    
    def log_http_response(self, status_code: int, headers: Optional[Dict] = None, body: Optional[Any] = None, duration: Optional[float] = None):
        extra = {
            "http_status": status_code,
            "http_headers": self._safe_serialize(headers) if headers else None,
            "http_body": self._safe_serialize(body) if body else None,
            "http_duration_seconds": f"{duration:.3f}" if duration else None
        }
        status_icon = "✅" if 200 <= status_code < 300 else "❌"
        self.log_debug(f"{status_icon} HTTP RESPONSE {status_code}", extra)
    
    def log_ai_request(self, prompt: str, model: str, max_tokens: Optional[int] = None):
        extra = {
            "ai_model": model,
            "ai_prompt_preview": prompt[:500] + ("..." if len(prompt) > 500 else ""),
            "ai_prompt_length": len(prompt),
            "ai_max_tokens": max_tokens
        }
        self.log_debug("AI REQUEST", extra)
    
    def log_ai_response(self, response: Any, duration: float, success: bool = True):
        extra = {
            "ai_response": self._safe_serialize(response),
            "ai_duration_seconds": f"{duration:.3f}",
            "ai_success": success
        }
        icon = "✅" if success else "❌"
        self.log_debug(f"{icon} AI RESPONSE", extra)
    
    def log_database_operation(self, operation: str, table: str, data: Optional[Any] = None, duration: Optional[float] = None):
        """Log database operation details."""
        extra = {
            "db_operation": operation,
            "db_table": table,
            "db_data": self._safe_serialize(data) if data else None,
            "db_duration_seconds": f"{duration:.3f}" if duration else None
        }
        self.log_debug("DATABASE OPERATION", extra)

# Global instance
_logger = UltraRobustDebugLogger()

# Public API functions
def log_debug(message: str, extra_data: Optional[Dict] = None):
    _logger.log_debug(message, extra_data)

def log_error(message: str, exception: Optional[Exception] = None):
    _logger.log_error(message, exception)

def trace_function(log_args: bool = True, log_result: bool = True, log_time: bool = True):
    return _logger.trace_function(log_args, log_result, log_time)

def log_http_request(method: str, url: str, headers: Optional[Dict] = None, body: Optional[Any] = None):
    _logger.log_http_request(method, url, headers, body)

def log_http_response(status_code: int, headers: Optional[Dict] = None, body: Optional[Any] = None, duration: Optional[float] = None):
    _logger.log_http_response(status_code, headers, body, duration)

def log_ai_request(prompt: str, model: str, max_tokens: Optional[int] = None):
    _logger.log_ai_request(prompt, model, max_tokens)

def log_ai_response(response: Any, duration: float, success: bool = True):
    _logger.log_ai_response(response, duration, success)

def log_database_operation(operation: str, table: str, data: Optional[Any] = None, duration: Optional[float] = None):
    _logger.log_database_operation(operation, table, data, duration)

if __name__ == "__main__":
    log_debug("STARTING ULTRA ROBUST DEBUG LOGGER TEST")
    
    @trace_function(log_args=True, log_result=True, log_time=True)
    def test_function(x, y=10, **kwargs):
        log_debug("Inside test_function", {"x": x, "y": y, "kwargs": kwargs})
        if isinstance(x, str):
            raise ValueError("Strings not allowed in this test")
        result = x + y
        time.sleep(0.1)
        return result
    
    try:
        test_function(5, y=3, extra_param="hello")
        test_function(100, 200)
        test_function("hello", y=20)
    except Exception as e:
        log_error("Expected test error occurred", e)
    
    log_http_request("POST", "https://api.example.com/data", {"Authorization": "Bearer token"}, {"key": "value"})
    log_http_response(200, {"Content-Type": "application/json"}, {"result": "success"}, 0.45)
    log_ai_request("Classify this text...", "llama3.2:3b", 200)
    log_ai_response({"category": "TEST", "confidence": 0.95}, 2.34, True)
    log_database_operation("INSERT", "tickets", {"id": 123, "message": "test"}, 0.012)
    log_debug("🧪 ULTRA ROBUST DEBUG LOGGER TEST COMPLETED")
    
    if _logger.log_file and _logger.log_file.exists():
        print(f"Log file created at: {_logger.log_file.absolute()}")
    else:
        print("Log file not created (console-only mode)")
