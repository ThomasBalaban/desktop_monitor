"""
Configuration settings for Desktop Monitor
"""
import os
import torch # type: ignore

class Config:
    """Configuration class for Desktop Monitor"""
    
    # Audio Configuration
    FS = 16000  # Sample rate
    CHUNK_DURATION = 3.0  # Duration of audio chunks in seconds
    OVERLAP = 0.5  # Overlap between chunks in seconds
    MAX_THREADS = 4  # Maximum number of processing threads
    
    # Audio file storage
    SAVE_DIR = "audio_captures"
    KEEP_AUDIO_FILES = False  # Set to True to keep audio files for debugging
    
    # Whisper model configuration
    MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
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
    MIN_FRAME_CHANGE = 0.10  # Minimum change threshold to process new frame
    SUMMARY_INTERVAL = 30  # Generate summary every N seconds
    FRAME_RESIZE = (800, 600)  # Resize frames for processing
    
    # YOLO Configuration
    YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)
    YOLO_CONFIDENCE = 0.5  # Confidence threshold for detections (0.0-1.0)
    YOLO_FPS = 2.0  # Target FPS for YOLO processing (lower = less CPU usage)
    YOLO_INPUT_SIZE = 640  # Max input size for YOLO (smaller = faster)
    YOLO_ENABLE_TRACKING = True  # Enable object tracking across frames
    YOLO_LOG_HIGH_CONFIDENCE = 0.8  # Log detections above this confidence
    
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
                print("Using default screen resolution: 1920x1080")
        
        print(f"Configuration loaded:")
        print(f"  Audio: Whisper {self.MODEL_SIZE} on {self.DEVICE}")
        print(f"  LLM Vision: {self.VISION_MODEL}")
        if self.ENABLE_YOLO_MONITORING:
            print(f"  YOLO: {self.YOLO_MODEL} (confidence: {self.YOLO_CONFIDENCE}, fps: {self.YOLO_FPS})")
        print(f"  Monitor area: {self.MONITOR_AREA}")