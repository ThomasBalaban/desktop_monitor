"""
Enhanced Configuration settings for Desktop Monitor with Advanced YOLO Summaries
"""
import os
import torch # type: ignore

class Config:
    """Enhanced Configuration class for Desktop Monitor"""
    
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
    
    # Vision Configuration
    ENABLE_VISION_MONITORING = True
    ENABLE_AUDIO_MONITORING = True
    ENABLE_YOLO_MONITORING = True  # Enable YOLO object detection
    
    # Monitor area (left, top, width, height) - captures entire screen by default
    # You can customize this to capture specific areas
    MONITOR_AREA = {"left": 16, "top": 157, "width": 1220, "height": 686}
    
    # LLM Vision models
    VISION_MODEL = "qwen2.5vl:7b"
    SUMMARY_MODEL = "mistral-nemo:latest"
    
    # LLM Vision processing
    MIN_FRAME_CHANGE = 0.12  # Minimum change threshold to process new frame
    FRAME_RESIZE = (800, 600)  # Larger frame size for better detail recognition
    SUMMARY_INTERVALSUMMARY_INTERVAL = 10  # Generate summary every N seconds
    
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
    YOLO_ENHANCED_LOGGING = True  # Enable enhanced logging with spatial info
    
    # Activity Detection Configuration
    YOLO_ACTIVITY_DETECTION = True  # Enable activity pattern detection
    YOLO_ACTIVITY_HISTORY_FRAMES = 10  # Number of frames to analyze for patterns
    YOLO_MOVEMENT_THRESHOLD = 50  # Pixel distance for rapid movement detection
    YOLO_WORKSPACE_DISTANCE = 300  # Distance to consider person "near" workspace
    YOLO_LOG_ACTIVITIES = True  # Log detected activities separately
    
    # Integration Settings
    YOLO_TRIGGER_LLM = True  # Use YOLO detections to trigger LLM vision analysis
    YOLO_LLM_TRIGGER_CLASSES = [  # Classes that should trigger LLM analysis
        'person', 'tv', 'laptop', 'cell phone', 'book', 'mouse', 'keyboard'
    ]
    YOLO_LLM_TRIGGER_CONFIDENCE = 0.7  # Minimum confidence to trigger LLM
    
    # Ollama configuration
    OLLAMA_GPU_LAYERS = "99"
    OLLAMA_KEEP_ALIVE = "0"
    
    def __init__(self):
        """Initialize configuration and create necessary directories"""
        # Set environment variables
        os.environ['OLLAMA_GPU_LAYERS'] = self.OLLAMA_GPU_LAYERS
        os.environ['OLLAMA_GGML_METAL'] = '1'
        os.environ['OLLAMA_KEEP_ALIVE'] = self.OLLAMA_KEEP_ALIVE
        
        # Reduce OpenCV verbose output
        os.environ['CV_IMPORT_VERBOSE'] = '0'
        os.environ['CV_IO_SUPPRESS_MSGF'] = '1'
        os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
        
        # Create audio capture directory
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        
        # Auto-detect screen resolution if not set
        if self.MONITOR_AREA["width"] == 1920 and self.MONITOR_AREA["height"] == 1080:
            try:
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
                print(f"Auto-detected screen resolution: {screen_width}x{screen_height}")
            except Exception:
                print("Using default screen resolution: 1220x686")
        
        # Validate YOLO summary style
        valid_styles = ['basic', 'spatial', 'confidence', 'scene', 'prominence', 'temporal']
        if self.YOLO_SUMMARY_STYLE not in valid_styles:
            print(f"Warning: Invalid YOLO_SUMMARY_STYLE '{self.YOLO_SUMMARY_STYLE}'. Using 'scene'.")
            self.YOLO_SUMMARY_STYLE = 'scene'
        
        print(f"Enhanced Configuration loaded:")
        print(f"  Audio: Whisper {self.MODEL_SIZE} on {self.DEVICE}")
        print(f"  LLM Vision: {self.VISION_MODEL}")
        if self.ENABLE_YOLO_MONITORING:
            print(f"  Enhanced YOLO: {self.YOLO_MODEL} (confidence: {self.YOLO_CONFIDENCE}, fps: {self.YOLO_FPS})")
            print(f"  YOLO Summary Style: {self.YOLO_SUMMARY_STYLE}")
            if self.YOLO_VERBOSE_SUMMARIES:
                print(f"  YOLO Verbose Summaries: enabled")
            if self.YOLO_SUMMARY_ROTATION:
                print(f"  YOLO Summary Rotation: enabled")
        print(f"  Monitor area: {self.MONITOR_AREA}")
    
    def set_yolo_summary_style(self, style: str) -> bool:
        """Dynamically change YOLO summary style"""
        valid_styles = ['basic', 'spatial', 'confidence', 'scene', 'prominence', 'temporal']
        if style in valid_styles:
            self.YOLO_SUMMARY_STYLE = style
            print(f"YOLO summary style changed to: {style}")
            return True
        else:
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