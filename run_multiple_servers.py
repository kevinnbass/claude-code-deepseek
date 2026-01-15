#!/usr/bin/env python3
"""
Script to run multiple instances of the GLM proxy server on different ports.
This allows you to have multiple isolated server instances running simultaneously.
"""

import argparse
import subprocess
import sys
import time
import signal
import os
from typing import List, Dict
import json

class ServerManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.server_configs: List[Dict] = []

    def start_server(self, host: str, port: int, instance_id: str = None, always_cot: bool = False):
        """Start a single server instance."""
        cmd = [sys.executable, "server.py"]

        if always_cot:
            cmd.append("--always-cot")

        cmd.extend(["--host", host, "--port", str(port)])

        if instance_id:
            cmd.extend(["--instance-id", instance_id])

        print(f"Starting server: {' '.join(cmd)}")

        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )

        self.processes.append(process)
        self.server_configs.append({
            "host": host,
            "port": port,
            "instance_id": instance_id,
            "always_cot": always_cot,
            "process": process
        })

        return process

    def stop_all(self):
        """Stop all running server instances."""
        print("\nStopping all servers...")
        for config in self.server_configs:
            process = config["process"]
            if process.poll() is None:  # Process is still running
                print(f"Stopping server on {config['host']}:{config['port']}")
                process.terminate()

        # Wait for processes to terminate
        for config in self.server_configs:
            process = config["process"]
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing server on {config['host']}:{config['port']}")
                process.kill()

        self.processes.clear()
        self.server_configs.clear()

    def list_servers(self):
        """List all running server instances."""
        print("\nRunning servers:")
        for i, config in enumerate(self.server_configs):
            process = config["process"]
            status = "Running" if process.poll() is None else f"Stopped (exit code: {process.returncode})"
            instance_id = f" [{config['instance_id']}]" if config['instance_id'] else ""
            print(f"  {i+1}. {config['host']}:{config['port']}{instance_id} - {status}")

def load_config(config_file: str) -> List[Dict]:
    """Load server configurations from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get('servers', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        return []

def create_default_config():
    """Create a default configuration for multiple servers."""
    return [
        {"host": "127.0.0.1", "port": 8082, "instance_id": "server1", "always_cot": True},
        {"host": "127.0.0.1", "port": 8083, "instance_id": "server2", "always_cot": True},
        {"host": "127.0.0.1", "port": 8084, "instance_id": "server3", "always_cot": True}
    ]

def main():
    parser = argparse.ArgumentParser(description="Run multiple GLM proxy server instances")
    parser.add_argument("--config", type=str, help="JSON configuration file for servers")
    parser.add_argument("--count", type=int, default=3, help="Number of servers to start (default: 3)")
    parser.add_argument("--base-port", type=int, default=8082, help="Base port number (default: 8082)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind servers to (default: 127.0.0.1)")
    parser.add_argument("--always-cot", action="store_true", help="Enable Chain-of-Thought for all servers")
    parser.add_argument("--create-config", type=str, help="Create a default configuration file and exit")
    parser.add_argument("--list", action="store_true", help="List running servers and exit")

    args = parser.parse_args()

    # Handle special commands
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, 'w') as f:
            json.dump({"servers": config}, f, indent=2)
        print(f"Created configuration file: {args.create_config}")
        print("Edit this file to customize your server configurations.")
        return

    manager = ServerManager()

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived signal, shutting down...")
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load configuration or create default
        if args.config:
            servers_config = load_config(args.config)
        else:
            # Create default configuration based on command line args
            servers_config = []
            for i in range(args.count):
                servers_config.append({
                    "host": args.host,
                    "port": args.base_port + i,
                    "instance_id": f"server{i+1}",
                    "always_cot": args.always_cot
                })

        if not servers_config:
            print("No server configurations found. Use --create-config to create a template.")
            return

        # Start all servers
        print(f"Starting {len(servers_config)} server instances...")
        for config in servers_config:
            manager.start_server(**config)
            time.sleep(0.5)  # Small delay between starts

        if args.list:
            manager.list_servers()
            return

        # Wait for user input or signal
        print("\nAll servers started! Press Ctrl+C to stop all servers.")
        print("Use 'ANTHROPIC_BASE_URL=http://127.0.0.1:<PORT> claude' to connect to each server.")

        # Keep the main process alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    finally:
        manager.stop_all()

if __name__ == "__main__":
    main()