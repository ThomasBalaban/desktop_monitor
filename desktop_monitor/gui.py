#!/usr/bin/env python3
"""
Desktop Monitor GUI - macOS Interface

A modern tkinter-based GUI for the Desktop Monitor application.
Provides real-time monitoring displays and system controls.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime, timedelta
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import sys
import io
import base64

# Import from same package
from .audio_system import DesktopAudioMonitor
from .vision_system import DesktopVisionMonitor
from .yolo_vision_system import DesktopYOLOMonitor
from .config import Config

class DesktopMonitorGUI:
    """Main GUI application for Desktop Monitor"""
    
    def __init__(self):
        self.config = Config()
        
        # Monitoring systems
        self.audio_monitor = None
        self.vision_monitor = None
        self.yolo_monitor = None
        self.monitoring_active = False
        
        # UI state
        self.last_summary_time = None
        self.last_summary_text = "No summary generated yet"
        self.current_image = None
        
        # Data storage for UI updates
        self.latest_audio = "No audio detected"
        self.latest_yolo = "No objects detected"
        self.latest_visual_frame = None
        self.current_audio_type = "speech"  # Track current audio classification
        
        # Setup main window
        self.setup_main_window()
        self.setup_ui_components()
        
        # Start UI update loop
        self.update_ui_loop()
        
    def setup_main_window(self):
        """Initialize the main window"""
        self.root = tk.Tk()
        self.root.title("Desktop Monitor")
        self.root.geometry("1200x800")
        
        # macOS specific styling
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom colors for macOS look
        self.style.configure('Title.TLabel', 
                           font=('SF Pro Display', 16, 'bold'),
                           background='#f0f0f0')
        self.style.configure('Heading.TLabel', 
                           font=('SF Pro Display', 12, 'bold'),
                           background='#f0f0f0')
        self.style.configure('Info.TLabel', 
                           font=('SF Pro Display', 10),
                           background='#f0f0f0')
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui_components(self):
        """Setup all UI components"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and controls
        self.setup_header(main_frame)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel for monitoring displays
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_frame.configure(width=350)
        left_frame.pack_propagate(False)
        
        # Right panel for summary
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
        
    def setup_header(self, parent):
        """Setup header with title and controls"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="Desktop Monitor", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(header_frame, text="● Stopped", 
                                    foreground='red', style='Info.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.start_button = ttk.Button(button_frame, text="Start Monitoring", 
                                     command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Monitoring", 
                                    command=self.stop_monitoring, state='disabled')
        self.stop_button.pack(side=tk.LEFT)
        
    def setup_left_panel(self, parent):
        """Setup left panel with monitoring displays"""
        # Audio section
        audio_header_frame = ttk.Frame(parent)
        audio_header_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Audio label with classification indicator
        self.audio_frame_label = ttk.Label(audio_header_frame, text="Audio Transcription", 
                                         font=('SF Pro Display', 11, 'bold'))
        self.audio_frame_label.pack(side=tk.LEFT)
        
        self.audio_type_label = ttk.Label(audio_header_frame, text="[SPEECH]", 
                                        foreground='blue', font=('SF Pro Display', 9, 'bold'))
        self.audio_type_label.pack(side=tk.LEFT, padx=(10, 0))
        
        audio_frame = ttk.LabelFrame(parent, text="", padding=10)
        audio_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.audio_text = scrolledtext.ScrolledText(audio_frame, height=6, width=40,
                                                  font=('SF Mono', 10), wrap=tk.WORD)
        self.audio_text.pack(fill=tk.BOTH, expand=True)
        self.audio_text.insert(tk.END, "Audio monitoring not started")
        self.audio_text.configure(state='disabled')
        
        # Visual section
        visual_frame = ttk.LabelFrame(parent, text="Current Screen", padding=10)
        visual_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Image display
        self.image_label = ttk.Label(visual_frame, text="No image captured")
        self.image_label.pack()
        
        # YOLO section
        yolo_frame = ttk.LabelFrame(parent, text="Object Detection", padding=10)
        yolo_frame.pack(fill=tk.BOTH, expand=True)
        
        self.yolo_text = scrolledtext.ScrolledText(yolo_frame, height=8, width=40,
                                                 font=('SF Mono', 10), wrap=tk.WORD)
        self.yolo_text.pack(fill=tk.BOTH, expand=True)
        self.yolo_text.insert(tk.END, "Object detection not started")
        self.yolo_text.configure(state='disabled')
        
    def setup_right_panel(self, parent):
        """Setup right panel with summary display"""
        # Summary header
        summary_header = ttk.Frame(parent)
        summary_header.pack(fill=tk.X, pady=(0, 10))
        
        summary_title = ttk.Label(summary_header, text="Activity Summary", style='Heading.TLabel')
        summary_title.pack(side=tk.LEFT)
        
        self.summary_time_label = ttk.Label(summary_header, text="No summary yet", 
                                          style='Info.TLabel')
        self.summary_time_label.pack(side=tk.RIGHT)
        
        # Summary content
        summary_frame = ttk.LabelFrame(parent, text="", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, 
                                                    font=('SF Pro Display', 11),
                                                    wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.insert(tk.END, "Activity summary will appear here once monitoring starts...")
        
    def start_monitoring(self):
        """Start the monitoring systems"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_label.configure(text="● Starting...", foreground='orange')
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=self._start_monitoring_systems, daemon=True)
        monitor_thread.start()
        
    def _start_monitoring_systems(self):
        """Start monitoring systems in background thread"""
        try:
            # Start audio monitoring
            if self.config.ENABLE_AUDIO_MONITORING:
                self.audio_monitor = DesktopAudioMonitor(self.config)
                self.audio_monitor.result_queue.put = self._audio_callback
                audio_thread = threading.Thread(target=self.audio_monitor.start, daemon=True)
                audio_thread.start()
            
            # Start vision monitoring
            if self.config.ENABLE_VISION_MONITORING:
                self.vision_monitor = DesktopVisionMonitor(self.config)
                # Override analysis result handling
                original_analyze = self.vision_monitor.processor.analyze_frame
                self.vision_monitor.processor.analyze_frame = self._vision_callback_wrapper(original_analyze)
                vision_thread = threading.Thread(target=self.vision_monitor.start, daemon=True)
                vision_thread.start()
            
            # Start YOLO monitoring
            if self.config.ENABLE_YOLO_MONITORING:
                self.yolo_monitor = DesktopYOLOMonitor(self.config)
                self.yolo_monitor.set_detection_callback(self._yolo_callback)
                yolo_thread = threading.Thread(target=self.yolo_monitor.start, daemon=True)
                yolo_thread.start()
            
            # Update status
            self.root.after(1000, lambda: self.status_label.configure(text="● Running", foreground='green'))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Failed to start monitoring: {e}"))
            self.monitoring_active = False
            self.root.after(0, self._reset_buttons)
    
    def stop_monitoring(self):
        """Stop the monitoring systems"""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        self.status_label.configure(text="● Stopping...", foreground='orange')
        
        # Stop monitoring systems
        if self.audio_monitor:
            self.audio_monitor.stop()
            self.audio_monitor = None
            
        if self.vision_monitor:
            self.vision_monitor.stop()
            self.vision_monitor = None
            
        if self.yolo_monitor:
            self.yolo_monitor.stop()
            self.yolo_monitor = None
        
        # Reset UI
        self._reset_buttons()
        self.status_label.configure(text="● Stopped", foreground='red')
        
        # Clear displays
        self._clear_displays()
        
    def _reset_buttons(self):
        """Reset button states"""
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        
    def _clear_displays(self):
        """Clear all display areas"""
        self.audio_text.configure(state='normal')
        self.audio_text.delete(1.0, tk.END)
        self.audio_text.insert(tk.END, "Audio monitoring stopped")
        self.audio_text.configure(state='disabled')
        
        # Reset audio type to default
        self.current_audio_type = "speech"
        self.audio_type_label.configure(text="[SPEECH]", foreground='blue')
        
        self.yolo_text.configure(state='normal')
        self.yolo_text.delete(1.0, tk.END)
        self.yolo_text.insert(tk.END, "Object detection stopped")
        self.yolo_text.configure(state='disabled')
        
        self.image_label.configure(image='', text="No image captured")
        
    def _audio_callback(self, audio_data):
        """Handle audio transcription results"""
        if isinstance(audio_data, tuple) and len(audio_data) >= 3:
            text, filename, audio_type, confidence = audio_data
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"[{timestamp}] [{audio_type.upper()} {confidence:.2f}] {text}\n"
            self.latest_audio = formatted_text.strip()
            
            # Update current audio type
            self.current_audio_type = audio_type.lower()
            
            # Update UI
            self.root.after(0, self._update_audio_display)
            self.root.after(0, self._update_audio_type_display)
    
    def _vision_callback_wrapper(self, original_analyze):
        """Wrapper for vision analysis to capture results"""
        def wrapper(frame):
            result, process_time = original_analyze(frame)
            
            # Capture the frame for display
            if isinstance(frame, Image.Image):
                self.latest_visual_frame = frame
            else:
                self.latest_visual_frame = Image.fromarray(frame)
            
            # Update summary
            self.last_summary_text = result
            self.last_summary_time = datetime.now()
            
            # Update UI
            self.root.after(0, self._update_vision_display)
            self.root.after(0, self._update_summary_display)
            
            return result, process_time
        return wrapper
    
    def _yolo_callback(self, result):
        """Handle YOLO detection results"""
        enhanced_summary = result.get('enhanced_summary', 'No objects detected')
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format YOLO output
        yolo_output = f"[{timestamp}] {enhanced_summary}\n"
        
        # Add activities if present
        analysis = result.get('analysis', {})
        activities = analysis.get('activities', [])
        if activities:
            yolo_output += f"Activities: {', '.join(activities)}\n"
        
        # Add changes if present
        changes = analysis.get('changes', [])
        if changes:
            yolo_output += f"Changes: {', '.join(changes)}\n"
        
        self.latest_yolo = yolo_output.strip()
        
        # Update UI
        self.root.after(0, self._update_yolo_display)
    
    def _update_audio_display(self):
        """Update audio display in UI thread"""
        self.audio_text.configure(state='normal')
        self.audio_text.insert(tk.END, self.latest_audio + "\n")
        self.audio_text.see(tk.END)
        
        # Keep only last 50 lines
        lines = self.audio_text.get(1.0, tk.END).split('\n')
        if len(lines) > 50:
            self.audio_text.delete(1.0, tk.END)
            self.audio_text.insert(1.0, '\n'.join(lines[-50:]))
            
        self.audio_text.configure(state='disabled')
    
    def _update_audio_type_display(self):
        """Update audio classification type display"""
        if self.current_audio_type == "speech":
            self.audio_type_label.configure(text="[SPEECH]", foreground='blue')
        elif self.current_audio_type == "music":
            self.audio_type_label.configure(text="[MUSIC]", foreground='green')
        else:
            self.audio_type_label.configure(text="[UNKNOWN]", foreground='gray')
    
    def _update_vision_display(self):
        """Update vision display in UI thread"""
        if self.latest_visual_frame:
            try:
                # Resize image for display
                display_image = self.latest_visual_frame.copy()
                display_image.thumbnail((300, 200), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
            except Exception as e:
                self.image_label.configure(image='', text=f"Image error: {e}")
    
    def _update_yolo_display(self):
        """Update YOLO display in UI thread"""
        self.yolo_text.configure(state='normal')
        self.yolo_text.insert(tk.END, self.latest_yolo + "\n\n")
        self.yolo_text.see(tk.END)
        
        # Keep only last 30 entries
        content = self.yolo_text.get(1.0, tk.END)
        entries = content.split('\n\n')
        if len(entries) > 30:
            self.yolo_text.delete(1.0, tk.END)
            self.yolo_text.insert(1.0, '\n\n'.join(entries[-30:]))
            
        self.yolo_text.configure(state='disabled')
    
    def _update_summary_display(self):
        """Update summary display in UI thread"""
        # Update summary text
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, self.last_summary_text)
        
        # Update time label
        if self.last_summary_time:
            time_diff = datetime.now() - self.last_summary_time
            seconds = int(time_diff.total_seconds())
            
            if seconds < 60:
                time_text = f"{seconds} seconds ago"
            elif seconds < 3600:
                minutes = seconds // 60
                time_text = f"{minutes} minutes ago"
            else:
                hours = seconds // 3600
                time_text = f"{hours} hours ago"
                
            self.summary_time_label.configure(text=time_text)
    
    def _show_error(self, message):
        """Show error message"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Error")
        error_window.geometry("400x150")
        error_window.configure(bg='#f0f0f0')
        
        error_label = ttk.Label(error_window, text=message, style='Info.TLabel')
        error_label.pack(expand=True, padx=20, pady=20)
        
        ok_button = ttk.Button(error_window, text="OK", command=error_window.destroy)
        ok_button.pack(pady=(0, 20))
    
    def update_ui_loop(self):
        """Continuous UI update loop"""
        # Update summary time every second
        if self.last_summary_time:
            self._update_summary_display()
            
        # Schedule next update
        self.root.after(1000, self.update_ui_loop)
    
    def on_closing(self):
        """Handle window closing"""
        if self.monitoring_active:
            self.stop_monitoring()
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        print("Starting Desktop Monitor GUI...")
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = DesktopMonitorGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()