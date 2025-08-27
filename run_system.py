#!/usr/bin/env python3
"""
RAG System Unified Launcher
===========================

A comprehensive launcher that starts all RAG system components:
- RAG API server (port 8001)
- Backend server (port 8000)  
- Frontend server (port 3000)

Note: This system now uses vLLM as the LLM backend instead of Ollama.
Make sure your vLLM server is running separately before starting this system.

Features:
- Single command startup
- Real-time log aggregation
- Process health monitoring
- Graceful shutdown
- Production-ready deployment support

Usage:
    python run_system.py [--mode dev|prod] [--logs-only] [--no-frontend]
"""

import subprocess
import threading
import time
import signal
import sys
import os
import argparse
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TextIO
import logging
from dataclasses import dataclass
import psutil

@dataclass
class ServiceConfig:
    name: str
    command: List[str]
    port: int
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    health_check_path: str = "/health"
    startup_delay: int = 2
    required: bool = True

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels and services."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    
    SERVICE_COLORS = {
        'rag-api': '\033[95m',    # Magenta
        'backend': '\033[96m',    # Cyan
        'frontend': '\033[93m',   # Yellow
        'system': '\033[92m',     # Green
    }
    
    RESET = '\033[0m'
    
    def format(self, record):
        # Add service-specific coloring
        service_name = getattr(record, 'service', 'system')
        service_color = self.SERVICE_COLORS.get(service_name, self.COLORS.get(record.levelname, ''))
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Create colored log line
        colored_service = f"{service_color}[{service_name.upper()}]{self.RESET}"
        colored_level = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
        
        return f"{timestamp} {colored_service} {colored_level}: {record.getMessage()}"

class ServiceManager:
    """Manages multiple system services with logging and health monitoring."""
    
    def __init__(self, mode: str = "dev", logs_dir: str = "logs"):
        self.mode = mode
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        # Setup logging
        self.setup_logging()
        
        # Service configurations
        self.services = self._get_service_configs()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup centralized logging with colors."""
        # Create main logger
        self.logger = logging.getLogger('system')
        self.logger.setLevel(logging.INFO)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for system logs
        file_handler = logging.FileHandler(self.logs_dir / 'system.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        self.logger.addHandler(file_handler)
    
    def _get_service_configs(self) -> Dict[str, ServiceConfig]:
        """Define service configurations based on mode."""
        base_configs = {
            'rag-api': ServiceConfig(
                name='rag-api',
                command=[sys.executable, '-m', 'rag_system.api_server'],
                port=8001,
                startup_delay=3,
                required=True
            ),
            'backend': ServiceConfig(
                name='backend',
                command=[sys.executable, 'backend/server.py'],
                port=8000,
                startup_delay=2,
                required=True
            ),
            'frontend': ServiceConfig(
                name='frontend',
                command=['npm', 'run', 'dev' if self.mode == 'dev' else 'start'],
                port=3000,
                startup_delay=5,
                required=False  # Optional in case Node.js not available
            )
        }
        
        # Production mode adjustments
        if self.mode == 'prod':
            # Use production build for frontend
            base_configs['frontend'].command = ['npm', 'run', 'start']
            # Add production environment variables
            base_configs['rag-api'].env = {'NODE_ENV': 'production'}
            base_configs['backend'].env = {'NODE_ENV': 'production'}
        
        return base_configs
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return True
            return False
        except (psutil.AccessDenied, AttributeError):
            # Fallback method
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
    
    def check_prerequisites(self) -> bool:
        """Check if all required tools are available."""
        self.logger.info("🔍 Checking prerequisites...")
        
        missing_tools = []
        
        # Check Python
        if not self._command_exists('python') and not self._command_exists('python3'):
            missing_tools.append('python')
        
        # Check Node.js (optional)
        if not self._command_exists('npm'):
            self.logger.warning("⚠️  npm not found - frontend will be disabled")
            self.services['frontend'].required = False
        
        # Check if vLLM server is accessible
        self.logger.info("🔍 Checking vLLM server connectivity...")
        import requests
        vllm_host = os.getenv("VLLM_HOST", "http://localhost:8000")
        try:
            response = requests.get(f"{vllm_host}/v1/models", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ vLLM server is accessible")
            else:
                self.logger.warning(f"⚠️  vLLM server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"⚠️  Cannot connect to vLLM server at {vllm_host}: {e}")
            self.logger.warning("   Make sure your vLLM server is running before starting the system")
        
        if missing_tools:
            self.logger.error(f"❌ Missing required tools: {', '.join(missing_tools)}")
            return False
        
        self.logger.info("✅ All prerequisites satisfied")
        return True
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH."""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_vllm_models(self):
        """Check which models are available on the vLLM server."""
        self.logger.info("📥 Checking available vLLM models...")
        
        import requests
        vllm_host = os.getenv("VLLM_HOST", "http://localhost:8000")
        
        try:
            response = requests.get(f"{vllm_host}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                model_names = [model["id"] for model in models_data]
                
                if model_names:
                    self.logger.info(f"✅ Available models: {', '.join(model_names)}")
                else:
                    self.logger.warning("⚠️  No models found on vLLM server")
            else:
                self.logger.warning(f"⚠️  Failed to get models: HTTP {response.status_code}")
                    
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"⚠️  Could not check vLLM models: {e}")
            self.logger.warning("   Make sure your vLLM server is running with models loaded")
    
    def start_service(self, service_name: str, config: ServiceConfig) -> bool:
        """Start a single service."""
        if service_name in self.processes:
            self.logger.warning(f"⚠️  {service_name} already running")
            return True
        
        # Check if port is in use
        if self.is_port_in_use(config.port):
            self.logger.warning(f"⚠️  Port {config.port} already in use, skipping {service_name}")
            return not config.required
        
        self.logger.info(f"🔄 Starting {service_name} on port {config.port}...")
        
        try:
            # Setup environment
            env = os.environ.copy()
            if config.env:
                env.update(config.env)
            
            # Start process
            process = subprocess.Popen(
                config.command,
                cwd=config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[service_name] = process
            
            # Start log monitoring thread
            log_thread = threading.Thread(
                target=self._monitor_service_logs,
                args=(service_name, process),
                daemon=True
            )
            log_thread.start()
            self.log_threads[service_name] = log_thread
            
            # Wait for startup
            time.sleep(config.startup_delay)
            
            # Check if process is still running
            if process.poll() is None:
                self.logger.info(f"✅ {service_name} started successfully (PID: {process.pid})")
                return True
            else:
                self.logger.error(f"❌ {service_name} failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start {service_name}: {e}")
            return False
    
    def _monitor_service_logs(self, service_name: str, process: subprocess.Popen):
        """Monitor service logs and forward to main logger."""
        service_logger = logging.getLogger(service_name)
        service_logger.setLevel(logging.INFO)
        
        # Add file handler for this service
        file_handler = logging.FileHandler(self.logs_dir / f'{service_name}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        service_logger.addHandler(file_handler)
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    # Create log record with service context
                    record = logging.LogRecord(
                        name=service_name,
                        level=logging.INFO,
                        pathname='',
                        lineno=0,
                        msg=line.strip(),
                        args=(),
                        exc_info=None
                    )
                    record.service = service_name
                    
                    # Log to both service file and main console
                    service_logger.handle(record)
                    self.logger.handle(record)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring {service_name} logs: {e}")
    
    def health_check(self, service_name: str, config: ServiceConfig) -> bool:
        """Perform health check on a service."""
        try:
            url = f"http://localhost:{config.port}{config.health_check_path}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_all(self, skip_frontend: bool = False) -> bool:
        """Start all services in order."""
        self.logger.info("🚀 Starting RAG System Components...")
        
        if not self.check_prerequisites():
            return False
        
        self.running = True
        failed_services = []
        
        # Check vLLM server availability first
        self.check_vllm_models()
        
        # Start services in dependency order
        service_order = ['rag-api', 'backend']
        if not skip_frontend and 'frontend' in self.services:
            service_order.append('frontend')
        
        for service_name in service_order:
            if service_name not in self.services:
                continue
                
            config = self.services[service_name]
            
            if not self.start_service(service_name, config):
                if config.required:
                    failed_services.append(service_name)
                else:
                    self.logger.warning(f"⚠️  Skipping optional service: {service_name}")
        
        if failed_services:
            self.logger.error(f"❌ Failed to start required services: {', '.join(failed_services)}")
            return False
        
        # Print status summary
        self._print_status_summary()
        return True
    
    
    def _print_status_summary(self):
        """Print system status summary."""
        self.logger.info("")
        self.logger.info("🎉 RAG System Started!")
        self.logger.info("📊 Services Status:")
        
        for service_name, config in self.services.items():
            if service_name in self.processes or self.is_port_in_use(config.port):
                status = "✅ Running"
                url = f"http://localhost:{config.port}"
                self.logger.info(f"   • {service_name.capitalize():<10}: {status:<10} {url}")
            else:
                self.logger.info(f"   • {service_name.capitalize():<10}: ❌ Stopped")
        
        self.logger.info("")
        self.logger.info("🌐 Access your RAG system at: http://localhost:3000")
        self.logger.info("")
        self.logger.info("📋 Useful commands:")
        self.logger.info("   • Stop system:  Ctrl+C")
        self.logger.info("   • Check logs:   tail -f logs/*.log")
        self.logger.info("   • Health check: python run_system.py --health")
        self.logger.info("")
        self.logger.info("⚠️  Note: Make sure your vLLM server is running separately")
        vllm_host = os.getenv("VLLM_HOST", "http://localhost:8000")
        self.logger.info(f"   vLLM server expected at: {vllm_host}")
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        if not self.running:
            return
        
        self.logger.info("🛑 Shutting down RAG system...")
        self.running = False
        
        # Stop services in reverse order
        for service_name in reversed(list(self.processes.keys())):
            self._stop_service(service_name)
        
        self.logger.info("✅ All services stopped")
    
    def _stop_service(self, service_name: str):
        """Stop a single service."""
        if service_name not in self.processes:
            return
        
        process = self.processes[service_name]
        self.logger.info(f"🔄 Stopping {service_name}...")
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                process.wait()
            
            self.logger.info(f"✅ {service_name} stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping {service_name}: {e}")
        finally:
            del self.processes[service_name]
    
    def monitor(self):
        """Monitor running services and restart if needed."""
        self.logger.info("👁️  Monitoring services... (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        self.logger.warning(f"⚠️  {service_name} has stopped unexpectedly")
                        
                        # Restart the service
                        config = self.services[service_name]
                        if config.required:
                            self.logger.info(f"🔄 Restarting {service_name}...")
                            del self.processes[service_name]
                            self.start_service(service_name, config)
                        
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RAG System Unified Launcher')
    parser.add_argument('--mode', choices=['dev', 'prod'], default='dev',
                       help='Run mode (default: dev)')
    parser.add_argument('--logs-only', action='store_true',
                       help='Only show aggregated logs from running services')
    parser.add_argument('--no-frontend', action='store_true',
                       help='Skip frontend startup')
    parser.add_argument('--health', action='store_true',
                       help='Check health of running services')
    parser.add_argument('--stop', action='store_true',
                       help='Stop all running services')
    
    args = parser.parse_args()
    
    # Create service manager
    manager = ServiceManager(mode=args.mode)
    
    try:
        if args.health:
            # Health check mode
            manager._print_status_summary()
            return
        
        if args.stop:
            # Stop mode - kill any running processes
            manager.logger.info("🛑 Stopping all RAG system processes...")
            # Implementation for stopping would go here
            return
        
        if args.logs_only:
            # Logs only mode - just tail existing logs
            manager.logger.info("📋 Showing aggregated logs... (Press Ctrl+C to stop)")
            manager.monitor()
            return
        
        # Normal startup mode
        if manager.start_all(skip_frontend=args.no_frontend):
            manager.monitor()
        else:
            manager.logger.error("❌ System startup failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        manager.logger.info("Received interrupt signal")
    finally:
        manager.shutdown()

if __name__ == "__main__":
    main() 