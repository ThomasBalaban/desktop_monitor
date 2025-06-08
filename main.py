#!/usr/bin/env python3
"""
Desktop Monitor - Standalone Desktop Audio and Visual Monitoring Application

This application monitors desktop audio and visual changes, providing real-time
transcription and analysis of what's happening on the screen.
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our modules
from desktop_monitor.audio_system import DesktopAudioMonitor
from desktop_monitor.vision_system import DesktopVisionMonitor
from desktop_monitor.config import Config

class DesktopMonitor:
    """Main desktop monitoring application"""
    
    def __init__(self):
        self.config = Config()
        self.audio_monitor = None
        self.vision_monitor = None
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\nShutting down Desktop Monitor...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the desktop monitoring system"""
        print("=" * 60)
        print("Desktop Monitor - Starting Up")
        print("=" * 60)
        
        self.running = True
        
        # Create audio monitor
        if self.config.ENABLE_AUDIO_MONITORING:
            print("Starting desktop audio monitoring...")
            self.audio_monitor = DesktopAudioMonitor(self.config)
            audio_thread = threading.Thread(target=self.audio_monitor.start, daemon=True)
            audio_thread.start()
            print("✓ Desktop audio monitoring started")
        
        # Create vision monitor
        if self.config.ENABLE_VISION_MONITORING:
            print("Starting desktop vision monitoring...")
            self.vision_monitor = DesktopVisionMonitor(self.config)
            vision_thread = threading.Thread(target=self.vision_monitor.start, daemon=True)
            vision_thread.start()
            print("✓ Desktop vision monitoring started")
        
        print("\n" + "=" * 60)
        print("Desktop Monitor is now running!")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Main loop
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the desktop monitoring system"""
        if not self.running:
            return
            
        print("\nStopping desktop monitoring...")
        self.running = False
        
        if self.audio_monitor:
            self.audio_monitor.stop()
            print("✓ Audio monitoring stopped")
        
        if self.vision_monitor:
            self.vision_monitor.stop()
            print("✓ Vision monitoring stopped")
        
        print("Desktop Monitor stopped successfully")

def main():
    """Main entry point"""
    try:
        monitor = DesktopMonitor()
        monitor.start()
    except Exception as e:
        print(f"Error starting Desktop Monitor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()