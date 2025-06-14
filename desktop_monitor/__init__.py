# desktop_monitor/__init__.py
"""
Desktop Monitor Package

A standalone application for monitoring desktop audio and visual activity.
Now includes YOLO object detection and a modern GUI interface.
"""

__version__ = "1.2.0"
__author__ = "Desktop Monitor"

from .config import Config
from .audio_system import DesktopAudioMonitor
from .vision_system import DesktopVisionMonitor
from .yolo_vision_system import DesktopYOLOMonitor
from .speech_music_classifier import SpeechMusicClassifier

# GUI is imported optionally to avoid tkinter dependency issues
try:
    from .gui import DesktopMonitorGUI
    __all__ = [
        'Config',
        'DesktopAudioMonitor', 
        'DesktopVisionMonitor',
        'DesktopYOLOMonitor',
        'SpeechMusicClassifier',
        'DesktopMonitorGUI'
    ]
except ImportError:
    # GUI not available (missing tkinter or PIL)
    __all__ = [
        'Config',
        'DesktopAudioMonitor', 
        'DesktopVisionMonitor',
        'DesktopYOLOMonitor',
        'SpeechMusicClassifier'
    ]