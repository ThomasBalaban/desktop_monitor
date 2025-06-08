# desktop_monitor/__init__.py
"""
Desktop Monitor Package

A standalone application for monitoring desktop audio and visual activity.
"""

__version__ = "1.0.0"
__author__ = "Desktop Monitor"

from .config import Config
from .audio_system import DesktopAudioMonitor
from .vision_system import DesktopVisionMonitor

__all__ = [
    'Config',
    'DesktopAudioMonitor', 
    'DesktopVisionMonitor'
]