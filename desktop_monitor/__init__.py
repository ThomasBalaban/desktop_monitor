# desktop_monitor/__init__.py
"""
Desktop Monitor Package

A standalone application for monitoring desktop audio and visual activity.
Now includes YOLO object detection for real-time object identification.
"""

__version__ = "1.1.0"
__author__ = "Desktop Monitor"

from .config import Config
from .audio_system import DesktopAudioMonitor
from .vision_system import DesktopVisionMonitor
from .yolo_vision_system import DesktopYOLOMonitor
from .speech_music_classifier import SpeechMusicClassifier

__all__ = [
    'Config',
    'DesktopAudioMonitor', 
    'DesktopVisionMonitor',
    'DesktopYOLOMonitor',
    'SpeechMusicClassifier'
]