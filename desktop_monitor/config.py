"""
Enhanced Configuration settings for Desktop Monitor with Burst Capture and Rolling Summaries

Optimized for macOS with burst capture, rolling summaries, and minimal logging.
"""
import os
import torch # type: ignore

class Config:
    """Enhanced Configuration class for Desktop Monitor with Burst Capture"""
    
    # Audio Configuration
    FS = 16000  # Sample rate
    CHUNK_DURATION = 2.0  # Duration of audio chunks in seconds
    OVERLAP = 0.5  # Overlap between chunks in seconds
    MAX_THREADS = 3  # Maximum number of processing threads
    
    # Audio file storage
    SAVE_DIR = "audio_captures"
    KEEP_AUDIO_FILES = False  # Set to True to keep audio files for debugging
    
    # Whisper model configuration
    MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Audio classification
    AUTO_DETECT_AUDIO_TYPE = True  # Automatically detect speech vs music
    DEBUG_CLASSIFIER = False  # Enable detailed classifier debugging
    
    # Classifier tuning parameters
    CLASSIFIER_HISTORY_SIZE = 5  # Number of classifications to consider for smoothing
    SPEECH_THRESHOLD = 0.6  # Fraction needed to classify as speech
    MUSIC_THRESHOLD = 0.4   # Fraction needed to classify as music
    
    # Core System Configuration
    ENABLE_VISION_MONITORING = True
    ENABLE_AUDIO_MONITORING = True
    ENABLE_YOLO_MONITORING = True
    
    # Burst Capture Configuration
    BURST_CAPTURE_ENABLED = True     # Enable burst capture for fast action detection
    BURST_FRAME_COUNT = 3            # Number of frames to capture in burst sequence
    BURST_FRAME_INTERVAL = 0.3       # Seconds between burst frames
    BURST_COOLDOWN = 2.0             # Minimum seconds between burst captures
    
    # Monitor area configuration (auto-detected for Mac)
    MONITOR_AREA = {"left": 15, "top": 155, "width": 1222, "height": 682}
    
    # Enhanced LLM Vision models (burst capture only)
    BURST_MODEL = "llava:7b"              # For burst frame analysis (LLaVA)
    SUMMARY_MODEL = "mistral-nemo:latest" # For rolling summaries (Nemo)
    
    # Enhanced LLM Vision processing
    MIN_FRAME_CHANGE = 0.10          # Minimum change threshold to process new frame
    SUMMARY_INTERVAL = 45            # Generate rolling summary every N seconds (increased for burst)
    FRAME_RESIZE = (800, 600)        # Resize frames for processing
    
    # Rolling Summary Configuration
    ROLLING_SUMMARY_ENABLED = True   # Enable rolling summary system
    ROLLING_SUMMARY_HISTORY = 5      # Keep last N rolling summaries
    ANALYSIS_HISTORY_SIZE = 15       # Keep last N individual analyses (increased for burst)
    ROLLING_SUMMARY_CONTEXT = 8      # Number of recent analyses to use for rolling summary
    
    # Mac-specific optimizations
    MACOS_OPTIMIZATIONS = True       # Enable Mac-specific optimizations
    MINIMAL_LOGGING = True           # Reduce console output (errors only)
    
    # Enhanced YOLO Configuration
    YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)
    YOLO_CONFIDENCE = 0.5  # Confidence threshold for detections (0.0-1.0)
    YOLO_FPS = 2.0  # Target FPS for YOLO processing (lower = less CPU usage)
    YOLO_INPUT_SIZE = 640  # Max input size for YOLO (smaller = faster)
    YOLO_ENABLE_TRACKING = True  # Enable object tracking across frames
    YOLO_LOG_HIGH_CONFIDENCE = 0.8  # Log detections above this confidence
    
    # Enhanced YOLO Summary Configuration
    YOLO_SUMMARY_STYLE = "scene"  # Options: basic, spatial, confidence, scene, prominence, temporal
    YOLO_VERBOSE_SUMMARIES = False  # Show all summary types in output
    YOLO_SUMMARY_ROTATION = False  # Rotate through different summary styles
    YOLO_ENHANCED_LOGGING = False   # Reduced logging for cleaner experience
    
    # Activity Detection Configuration
    YOLO_ACTIVITY_DETECTION = True  # Enable activity pattern detection
    YOLO_ACTIVITY_HISTORY_FRAMES = 10  # Number of frames to analyze for patterns
    YOLO_MOVEMENT_THRESHOLD = 50  # Pixel distance for rapid movement detection
    YOLO_WORKSPACE_DISTANCE = 300  # Distance to consider person "near" workspace
    YOLO_LOG_ACTIVITIES = False    # Reduced logging for cleaner experience
    
    # Integration Settings
    YOLO_TRIGGER_LLM = True  # Use YOLO detections to trigger LLM vision analysis
    YOLO_LLM_TRIGGER_CLASSES = [  # Classes that should trigger LLM analysis
        'person', 'tv', 'laptop', 'cell phone', 'book', 'mouse', 'keyboard'
    ]
    YOLO_LLM_TRIGGER_CONFIDENCE = 0.7  # Minimum confidence to trigger LLM
    
    # Ollama configuration (Mac optimized)
    OLLAMA_GPU_LAYERS = "99"
    OLLAMA_KEEP_ALIVE = "5m"         # Keep models loaded longer for burst capture
    OLLAMA_NUM_THREAD = "8"          # Optimize for Mac CPU cores
    OLLAMA_NUM_CTX = "2048"          # Context size for burst analysis
    
    # Performance tuning for burst capture
    BURST_PROCESSING_TIMEOUT = 10.0   # Max seconds to wait for burst processing
    VISION_PROCESSING_TIMEOUT = 8.0   # Max seconds to wait for single frame processing
    
    def __init__(self):
        """Enhanced initialization with Mac optimizations and burst capture support"""
        # Set environment variables for Mac optimization
        os.environ['OLLAMA_GPU_LAYERS'] = self.OLLAMA_GPU_LAYERS
        os.environ['OLLAMA_KEEP_ALIVE'] = self.OLLAMA_KEEP_ALIVE
        os.environ['OLLAMA_NUM_THREAD'] = self.OLLAMA_NUM_THREAD
        os.environ['OLLAMA_NUM_CTX'] = self.OLLAMA_NUM_CTX
        
        # Mac Metal optimization for Ollama
        os.environ['OLLAMA_GGML_METAL'] = '1'
        os.environ['OLLAMA_METAL'] = '1'
        
        # Reduce verbose output for cleaner experience
        if self.MINIMAL_LOGGING:
            os.environ['CV_IMPORT_VERBOSE'] = '0'
            os.environ['CV_IO_SUPPRESS_MSGF'] = '1'
            os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
            # Suppress TensorFlow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            # Suppress PyTorch warnings
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
        
        # Create audio capture directory
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        
        # Mac screen resolution auto-detection
        if self.MONITOR_AREA["width"] == 1920 and self.MONITOR_AREA["height"] == 1080:
            try:
                # Try Cocoa for more accurate Mac screen detection
                try:
                    from Cocoa import NSScreen # type: ignore
                    screens = NSScreen.screens()
                    if screens:
                        main_screen = screens[0]
                        frame = main_screen.frame()
                        self.MONITOR_AREA = {
                            "left": 0,
                            "top": 0,
                            "width": int(frame.size.width),
                            "height": int(frame.size.height)
                        }
                        if not self.MINIMAL_LOGGING:
                            print(f"Mac screen resolution detected: {int(frame.size.width)}x{int(frame.size.height)}")
                except ImportError:
                    # Fallback to tkinter if Cocoa not available
                    import tkinter as tk
                    root = tk.Tk()
                    screen_width = root.winfo_screenwidth()
                    screen_height = root.winfo_screenheight()
                    root.destroy()
                    
                    self.MONITOR_AREA = {
                        "left": 0,
                        "top": 0, 
                        "width": screen_width,
                        "height": screen_height
                    }
                    if not self.MINIMAL_LOGGING:
                        print(f"Screen resolution detected: {screen_width}x{screen_height}")
            except Exception:
                if not self.MINIMAL_LOGGING:
                    print("Using default screen resolution: 1920x1080")
        
        # Validate burst capture settings
        if self.BURST_CAPTURE_ENABLED:
            if self.BURST_FRAME_COUNT < 2:
                self.BURST_FRAME_COUNT = 2
            if self.BURST_FRAME_COUNT > 5:
                self.BURST_FRAME_COUNT = 5
            if self.BURST_FRAME_INTERVAL < 0.1:
                self.BURST_FRAME_INTERVAL = 0.1
            if self.BURST_FRAME_INTERVAL > 1.0:
                self.BURST_FRAME_INTERVAL = 1.0
            if self.BURST_COOLDOWN < 1.0:
                self.BURST_COOLDOWN = 1.0
        
        # Validate YOLO summary style
        valid_styles = ['basic', 'spatial', 'confidence', 'scene', 'prominence', 'temporal']
        if self.YOLO_SUMMARY_STYLE not in valid_styles:
            if not self.MINIMAL_LOGGING:
                print(f"Warning: Invalid YOLO_SUMMARY_STYLE '{self.YOLO_SUMMARY_STYLE}'. Using 'scene'.")
            self.YOLO_SUMMARY_STYLE = 'scene'
        
        # Print startup information (only if not minimal logging)
        if not self.MINIMAL_LOGGING:
            print(f"Enhanced Desktop Monitor Configuration loaded:")
            print(f"  Audio: Whisper {self.MODEL_SIZE} on {self.DEVICE}")
            if self.BURST_CAPTURE_ENABLED:
                print(f"  Burst Capture: {self.BURST_MODEL} ({self.BURST_FRAME_COUNT} frames @ {self.BURST_FRAME_INTERVAL}s)")
            print(f"  Rolling Summary: {self.SUMMARY_MODEL}")
            if self.ENABLE_YOLO_MONITORING:
                print(f"  YOLO: {self.YOLO_MODEL} (confidence: {self.YOLO_CONFIDENCE}, fps: {self.YOLO_FPS})")
                print(f"  YOLO Summary Style: {self.YOLO_SUMMARY_STYLE}")
            print(f"  Monitor area: {self.MONITOR_AREA}")
    
    def get_burst_settings(self) -> dict:
        """Get current burst capture settings"""
        return {
            'enabled': self.BURST_CAPTURE_ENABLED,
            'frame_count': self.BURST_FRAME_COUNT,
            'frame_interval': self.BURST_FRAME_INTERVAL,
            'cooldown': self.BURST_COOLDOWN,
            'model': self.BURST_MODEL,
            'timeout': self.BURST_PROCESSING_TIMEOUT
        }
    
    def get_rolling_summary_settings(self) -> dict:
        """Get current rolling summary settings"""
        return {
            'enabled': self.ROLLING_SUMMARY_ENABLED,
            'history_size': self.ROLLING_SUMMARY_HISTORY,
            'analysis_history': self.ANALYSIS_HISTORY_SIZE,
            'context_size': self.ROLLING_SUMMARY_CONTEXT,
            'interval': self.SUMMARY_INTERVAL,
            'model': self.SUMMARY_MODEL
        }
    
    def set_burst_enabled(self, enabled: bool):
        """Toggle burst capture on/off"""
        self.BURST_CAPTURE_ENABLED = enabled
        if not self.MINIMAL_LOGGING:
            status = "enabled" if enabled else "disabled"
            print(f"Burst capture {status}")
    
    def set_burst_sensitivity(self, sensitivity: str):
        """Set burst capture sensitivity preset"""
        if sensitivity == "high":
            self.MIN_FRAME_CHANGE = 0.05
            self.BURST_COOLDOWN = 1.0
            self.BURST_FRAME_INTERVAL = 0.2
        elif sensitivity == "medium":
            self.MIN_FRAME_CHANGE = 0.10
            self.BURST_COOLDOWN = 2.0
            self.BURST_FRAME_INTERVAL = 0.3
        elif sensitivity == "low":
            self.MIN_FRAME_CHANGE = 0.15
            self.BURST_COOLDOWN = 3.0
            self.BURST_FRAME_INTERVAL = 0.4
        
        if not self.MINIMAL_LOGGING:
            print(f"Burst sensitivity set to {sensitivity}")
    
    def set_gaming_mode(self, enabled: bool):
        """Enable/disable gaming optimizations"""
        if enabled:
            # Gaming mode: more sensitive, faster capture
            self.BURST_FRAME_COUNT = 3
            self.BURST_FRAME_INTERVAL = 0.2
            self.BURST_COOLDOWN = 1.5
            self.MIN_FRAME_CHANGE = 0.08
            self.SUMMARY_INTERVAL = 30  # More frequent summaries
        else:
            # Normal mode: balanced settings
            self.BURST_FRAME_COUNT = 3
            self.BURST_FRAME_INTERVAL = 0.3
            self.BURST_COOLDOWN = 2.0
            self.MIN_FRAME_CHANGE = 0.10
            self.SUMMARY_INTERVAL = 45
        
        if not self.MINIMAL_LOGGING:
            mode = "enabled" if enabled else "disabled"
            print(f"Gaming mode {mode}")
    
    def set_minimal_logging(self, enabled: bool):
        """Toggle minimal logging mode"""
        self.MINIMAL_LOGGING = enabled
        if enabled:
            os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        else:
            if 'PYTHONWARNINGS' in os.environ:
                del os.environ['PYTHONWARNINGS']
            if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
                del os.environ['TF_CPP_MIN_LOG_LEVEL']
    
    def set_yolo_summary_style(self, style: str) -> bool:
        """Dynamically change YOLO summary style"""
        valid_styles = ['basic', 'spatial', 'confidence', 'scene', 'prominence', 'temporal']
        if style in valid_styles:
            self.YOLO_SUMMARY_STYLE = style
            if not self.MINIMAL_LOGGING:
                print(f"YOLO summary style changed to: {style}")
            return True
        else:
            if not self.MINIMAL_LOGGING:
                print(f"Invalid style: {style}. Valid options: {', '.join(valid_styles)}")
            return False
    
    def get_summary_style_info(self) -> dict:
        """Get information about available summary styles"""
        return {
            'basic': 'Simple object counts (e.g., "2 persons, 1 laptop")',
            'spatial': 'Objects with screen positions (e.g., "person (center), laptop (top-left)")',
            'confidence': 'Objects with confidence levels (e.g., "person (0.85-high)")',
            'scene': 'Contextual scene analysis (e.g., "Scene: active_workspace | person using laptop")',
            'prominence': 'Objects by visual importance (e.g., "Dominated by: laptop (large-4.2%)")',
            'temporal': 'Changes over time (e.g., "Current: 2 persons | Changes: +1 person")'
        }
    
    def get_performance_settings(self) -> dict:
        """Get current performance-related settings"""
        return {
            'device': self.DEVICE,
            'max_threads': self.MAX_THREADS,
            'yolo_fps': self.YOLO_FPS,
            'frame_resize': self.FRAME_RESIZE,
            'burst_timeout': self.BURST_PROCESSING_TIMEOUT,
            'vision_timeout': self.VISION_PROCESSING_TIMEOUT,
            'minimal_logging': self.MINIMAL_LOGGING
        }
    
    def optimize_for_performance(self):
        """Apply performance optimizations for slower systems"""
        self.YOLO_FPS = 1.0
        self.BURST_FRAME_COUNT = 2
        self.BURST_COOLDOWN = 3.0
        self.FRAME_RESIZE = (640, 480)
        self.SUMMARY_INTERVAL = 60
        
        if not self.MINIMAL_LOGGING:
            print("Performance optimizations applied")