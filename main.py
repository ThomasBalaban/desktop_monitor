#!/usr/bin/env python3
"""
Desktop Monitor - Standalone Desktop Audio and Visual Monitoring Application

This application monitors desktop audio and visual changes, providing real-time
transcription and analysis of what's happening on the screen.

Now includes YOLO object detection for real-time object identification.
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
from desktop_monitor.yolo_vision_system import DesktopYOLOMonitor
from desktop_monitor.config import Config

class DesktopMonitor:
    """Main desktop monitoring application with multi-modal analysis"""
    
    def __init__(self):
        self.config = Config()
        self.audio_monitor = None
        self.vision_monitor = None
        self.yolo_monitor = None
        self.running = False
        
        # Integration state
        self.last_yolo_trigger = 0
        self.yolo_trigger_cooldown = 5.0  # Seconds between YOLO-triggered LLM analysis
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\nShutting down Desktop Monitor...")
        self.stop()
        sys.exit(0)
    
    def _yolo_detection_callback(self, result):
        """Handle YOLO detection results and potentially trigger LLM analysis"""
        if not self.config.YOLO_TRIGGER_LLM or not self.vision_monitor:
            return
            
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_yolo_trigger < self.yolo_trigger_cooldown:
            return
            
        # Check if any high-confidence detections match trigger classes
        detections = result.get('detections', [])
        trigger_detections = []
        
        for detection in detections:
            if (detection.confidence >= self.config.YOLO_LLM_TRIGGER_CONFIDENCE and
                detection.class_name in self.config.YOLO_LLM_TRIGGER_CLASSES):
                trigger_detections.append(detection)
        
        if trigger_detections:
            # Log the trigger
            trigger_objects = [f"{d.class_name}({d.confidence:.2f})" for d in trigger_detections]
            print(f"[INTEGRATION] YOLO triggered LLM analysis: {', '.join(trigger_objects)}")
            
            # Force LLM vision analysis by invalidating the last frame
            if hasattr(self.vision_monitor.processor, 'last_valid_frame'):
                self.vision_monitor.processor.last_valid_frame = None
                
            self.last_yolo_trigger = current_time
    
    def _print_system_status(self):
        """Print current status of all monitoring systems"""
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        
        if self.audio_monitor:
            print(f"🎵 Audio: Running | Active threads: {getattr(self.audio_monitor, 'active_threads', 'N/A')}")
            
        if self.vision_monitor:
            llm_status = "Running" if getattr(self.vision_monitor, 'running', False) else "Stopped"
            print(f"👁️  LLM Vision: {llm_status}")
            
        if self.yolo_monitor:
            yolo_objects = self.yolo_monitor.get_current_objects()
            print(f"🎯 YOLO: Running | Current objects: {yolo_objects}")
            recent_changes = self.yolo_monitor.get_recent_changes(seconds=60)
            if recent_changes:
                print(f"📋 Recent changes: {len(recent_changes)} in last 60s")
        
        print("="*60)
    
    def start(self):
        """Start the desktop monitoring system"""
        print("=" * 60)
        print("Desktop Monitor - Multi-Modal Monitoring System")
        print("=" * 60)
        
        self.running = True
        
        # Start audio monitoring
        if self.config.ENABLE_AUDIO_MONITORING:
            print("Starting desktop audio monitoring...")
            self.audio_monitor = DesktopAudioMonitor(self.config)
            audio_thread = threading.Thread(target=self.audio_monitor.start, daemon=True)
            audio_thread.start()
            print("✓ Desktop audio monitoring started")
        
        # Start LLM vision monitoring
        if self.config.ENABLE_VISION_MONITORING:
            print("Starting LLM vision monitoring...")
            self.vision_monitor = DesktopVisionMonitor(self.config)
            vision_thread = threading.Thread(target=self.vision_monitor.start, daemon=True)
            vision_thread.start()
            print("✓ LLM vision monitoring started")
        
        # Start YOLO monitoring
        if self.config.ENABLE_YOLO_MONITORING:
            print("Starting YOLO object detection...")
            self.yolo_monitor = DesktopYOLOMonitor(self.config)
            
            # Set up integration callback
            if self.config.YOLO_TRIGGER_LLM:
                self.yolo_monitor.set_detection_callback(self._yolo_detection_callback)
                
            yolo_thread = threading.Thread(target=self.yolo_monitor.start, daemon=True)
            yolo_thread.start()
            print("✓ YOLO object detection started")
        
        print("\n" + "=" * 60)
        print("🚀 Desktop Monitor is now running!")
        print("📊 Multi-modal monitoring active:")
        
        if self.config.ENABLE_AUDIO_MONITORING:
            print("   🎵 Audio transcription & classification")
        if self.config.ENABLE_VISION_MONITORING:
            print("   👁️  LLM-powered visual analysis") 
        if self.config.ENABLE_YOLO_MONITORING:
            print("   🎯 Real-time object detection")
        if self.config.YOLO_TRIGGER_LLM:
            print("   🔗 Integrated YOLO→LLM triggering")
            
        print("\nPress Ctrl+C to stop")
        print("=" * 60)
        
        # Status update loop
        last_status_update = time.time()
        status_interval = 60  # Print status every 60 seconds
        
        try:
            while self.running:
                time.sleep(1)
                
                # Periodic status update
                if time.time() - last_status_update > status_interval:
                    self._print_system_status()
                    last_status_update = time.time()
                    
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the desktop monitoring system"""
        if not self.running:
            return
            
        print("\n🛑 Stopping desktop monitoring...")
        self.running = False
        
        if self.audio_monitor:
            self.audio_monitor.stop()
            print("✓ Audio monitoring stopped")
        
        if self.vision_monitor:
            self.vision_monitor.stop()
            print("✓ LLM vision monitoring stopped")
            
        if self.yolo_monitor:
            self.yolo_monitor.stop()
            print("✓ YOLO monitoring stopped")
        
        print("🎯 Desktop Monitor stopped successfully")

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import whisper
    except ImportError:
        missing_deps.append("openai-whisper")
    
    try:
        import ollama
    except ImportError:
        missing_deps.append("ollama")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import sounddevice
    except ImportError:
        missing_deps.append("sounddevice")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main entry point"""
    print("🔍 Checking dependencies...")
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running.")
        sys.exit(1)
    
    try:
        monitor = DesktopMonitor()
        monitor.start()
    except Exception as e:
        print(f"❌ Error starting Desktop Monitor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()