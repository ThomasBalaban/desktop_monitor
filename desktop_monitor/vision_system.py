"""
Desktop Vision Monitoring System with Burst Capture

Enhanced version that captures multiple frames during fast action and maintains
rolling summaries for better context understanding.
"""

import cv2 # type: ignore
import time
import base64
import threading
import numpy as np # type: ignore
from mss import mss # type: ignore
from PIL import Image # type: ignore
from io import BytesIO
from collections import deque
from typing import List, Tuple, Optional

class VisionProcessor:
    """Enhanced processor with burst capture and rolling summary capabilities"""
    
    def __init__(self, config):
        self.config = config
        
        # Rolling analysis storage (increased from 10 to handle burst captures)
        self.analysis_history = deque(maxlen=15)
        self.rolling_summaries = deque(maxlen=5)  # Keep last 5 rolling summaries
        
        # Burst capture state
        self.burst_in_progress = False
        self.burst_lock = threading.Lock()
        
        # Frame validation
        self.last_valid_frame = None
        
        # Summary timing
        self.summary_lock = threading.Lock()
        self.last_summary_time = time.time()
        
        # Initialize Ollama models (burst-only)
        try:
            import ollama # type: ignore
            self.ollama = ollama
            
            # Warmup LLaVA for burst analysis (suppress output)
            try:
                self.ollama.generate(model="llava:7b", prompt="Ready", options={'num_predict': 1})
            except:
                print(f"[VISION ERROR] Could not load llava:7b model")
                raise
                
            # Warmup Nemo for summaries
            self.ollama.generate(model=config.SUMMARY_MODEL, prompt="Ready", options={'num_predict': 1})
            
        except Exception as e:
            print(f"[VISION ERROR] Failed to initialize Ollama: {e}")
            raise
            
        # Burst analysis prompt (updated for 5 frames)
        self.burst_analysis_prompt = """
        Analyze this sequence of 5 frames captured 0.3 seconds apart during detected screen activity.
        
        Describe what's happening across these frames in 2-3 sentences:
        - What action or activity is taking place
        - How the scene evolves through the sequence
        - The overall context (gaming, work, browsing, etc.)
        
        Focus on movement, transitions, and dynamic elements rather than static descriptions.
        """
        
    def optimize_frame(self, frame):
        """Convert PIL image to base64 JPEG for LLM processing"""
        frame = frame.resize(self.config.FRAME_RESIZE, Image.LANCZOS)
        buffered = BytesIO()
        frame.save(buffered, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def validate_frame(self, frame):
        """Check if frame has changed enough to warrant processing"""
        if self.last_valid_frame is None:
            return True
            
        current_np = np.array(frame)
        last_np = np.array(self.last_valid_frame)
        
        if current_np.shape != last_np.shape:
            return True
        
        diff = cv2.absdiff(current_np, last_np)
        change_percent = np.count_nonzero(diff) / diff.size
        
        return change_percent > self.config.MIN_FRAME_CHANGE
    
    def capture_burst_sequence(self, initial_frame: Image.Image, sct) -> List[Image.Image]:
        """Capture a sequence of 5 frames spaced 0.3 seconds apart"""
        frames = [initial_frame]
        
        for i in range(4):  # Capture 4 more frames (total of 5)
            time.sleep(0.3)
            
            # Capture next frame
            sct_img = sct.grab(self.config.MONITOR_AREA)
            frame = Image.frombytes(
                'RGB', 
                (sct_img.width, sct_img.height), 
                sct_img.rgb
            )
            frames.append(frame)
        
        return frames
    
    def analyze_frame(self, frames_list) -> Tuple[str, float]:
        """
        Analyze burst sequence only (no single frame analysis)
        
        Args:
            frames_list: List of PIL Images from burst capture
            
        Returns:
            Tuple of (analysis_result, process_time)
        """
        start_time = time.time()
        
        try:
            # Only burst analysis supported
            if not isinstance(frames_list, list):
                raise ValueError("Only burst analysis supported - frames must be a list")
                
            return self._analyze_burst_frames(frames_list, start_time)
                
        except Exception as e:
            error_msg = f"Burst analysis error: {str(e)}"
            print(f"[VISION ERROR] {error_msg}")
            return error_msg, time.time() - start_time
    
    def _analyze_burst_frames(self, frames: List[Image.Image], start_time: float) -> Tuple[str, float]:
        """Analyze a burst sequence of frames using LLaVA 7B"""
        try:
            # Prepare images for LLaVA
            frame_images = [self.optimize_frame(frame) for frame in frames]
            
            # Use LLaVA 7B for multi-frame analysis
            response = self.ollama.chat(
                model="llava:7b",
                messages=[{
                    "role": "user", 
                    "content": self.burst_analysis_prompt, 
                    "images": frame_images
                }],
                options={
                    'temperature': 0.3,
                    'num_ctx': 2048,
                    'num_predict': 150
                }
            )
            
            result = response['message']['content'].strip()
            process_time = time.time() - start_time
            
            # Store in analysis history
            self.analysis_history.append({
                'type': 'burst',
                'result': result,
                'timestamp': time.time(),
                'frames': len(frames),
                'model': "llava:7b"
            })
            
            return result, process_time
            
        except Exception as e:
            raise Exception(f"Burst frame analysis failed: {str(e)}")
    
    def generate_rolling_summary(self):
        """Generate enhanced rolling summary from recent burst analyses"""
        with self.summary_lock:
            recent_analyses = list(self.analysis_history)[-8:]  # Get more context

        if len(recent_analyses) < 2:
            return "Insufficient activity data"

        # All analyses are burst analyses now
        burst_analyses = [a for a in recent_analyses if a['type'] == 'burst']
        
        # Build context for summary
        if burst_analyses:
            burst_context = " | ".join([a['result'] for a in burst_analyses[-3:]])
            context_part = f"Recent dynamic activity: {burst_context}"
        else:
            return "No recent burst activity to summarize"
        
        # Enhanced summary prompt
        summary_prompt = f"""
        Analyze this sequence of desktop monitoring burst capture observations and provide a rolling summary.
        
        {context_part}
        
        Provide a 2-3 sentence summary that:
        - Identifies the primary activity or context (gaming, work, browsing, etc.)
        - Describes the overall progression or changes observed
        - Captures the current state or focus
        
        Be specific about what the user appears to be doing rather than just describing interface elements.
        """
        
        try:
            response = self.ollama.generate(
                model=self.config.SUMMARY_MODEL,
                prompt=summary_prompt,
                options={
                    'temperature': 0.3, 
                    'num_predict': 120,
                    'num_ctx': 2048
                }
            )
            
            summary_text = response['response'].strip()
            
            # Store the rolling summary
            with self.summary_lock:
                self.rolling_summaries.append({
                    'summary': summary_text,
                    'timestamp': time.time(),
                    'analysis_count': len(recent_analyses),
                    'burst_count': len(burst_analyses)
                })
            
            return summary_text
            
        except Exception as e:
            error_msg = f"Rolling summary error: {str(e)}"
            print(f"[VISION ERROR] {error_msg}")
            return error_msg
    
    def get_latest_rolling_summary(self) -> str:
        """Get the most recent rolling summary"""
        with self.summary_lock:
            if self.rolling_summaries:
                return self.rolling_summaries[-1]['summary']
            return "No rolling summary available yet"
    
    def summary_worker(self):
        """Background worker to generate periodic rolling summaries"""
        while True:
            time.sleep(1)
            if time.time() - self.last_summary_time > self.config.SUMMARY_INTERVAL:
                self.generate_rolling_summary()
                self.last_summary_time = time.time()

class DesktopVisionMonitor:
    """Enhanced desktop vision monitoring with burst capture"""
    
    def __init__(self, config):
        self.config = config
        self.processor = VisionProcessor(config)
        self.running = False
        
        # Burst capture state
        self.last_burst_time = 0
        self.burst_cooldown = 2.0  # Minimum seconds between bursts
        
        # Set up OpenCV (minimal configuration for Mac)
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
        
    def start(self):
        """Start enhanced desktop vision monitoring"""
        self.running = True
        
        # Start summary worker thread
        summary_thread = threading.Thread(target=self.processor.summary_worker, daemon=True)
        summary_thread.start()
        
        try:
            with mss() as sct:
                while self.running:
                    try:
                        # Capture initial frame
                        sct_img = sct.grab(self.config.MONITOR_AREA)
                        frame = Image.frombytes(
                            'RGB', 
                            (sct_img.width, sct_img.height), 
                            sct_img.rgb
                        )
                        
                        # Check if frame changed enough to process
                        if self.processor.validate_frame(frame):
                            self.processor.last_valid_frame = frame.copy()
                            current_time = time.time()
                            
                            # Decide whether to do burst capture or single frame analysis
                            if (current_time - self.last_burst_time > self.burst_cooldown and 
                                not self.processor.burst_in_progress):
                                
                                # Initiate burst capture
                                with self.processor.burst_lock:
                                    self.processor.burst_in_progress = True
                                    self.last_burst_time = current_time
                                
                                try:
                                    # Capture burst sequence
                                    burst_frames = self.processor.capture_burst_sequence(frame, sct)
                                    
                                    # Analyze the burst
                                    result, process_time = self.processor.analyze_frame(burst_frames)
                                    
                                    # For GUI: only show the first frame from the burst
                                    if hasattr(self.processor, '_gui_callback'):
                                        self.processor._gui_callback(burst_frames[0], result, process_time)
                                    
                                finally:
                                    with self.processor.burst_lock:
                                        self.processor.burst_in_progress = False
                            else:
                                # Single frame analysis (when in cooldown or burst in progress)
                                result, process_time = self.processor.analyze_frame(frame)
                                
                                # For GUI callback
                                if hasattr(self.processor, '_gui_callback'):
                                    self.processor._gui_callback(frame, result, process_time)
                        else:
                            # Small sleep when no changes detected
                            time.sleep(0.1)

                        # Exit check (removed keyboard check for Mac compatibility)
                        
                    except Exception as e:
                        print(f"[VISION ERROR] Frame processing: {str(e)}")
                        time.sleep(1)
                        
        except Exception as e:
            print(f"[VISION ERROR] Monitoring failed: {e}")
        finally:
            self.running = False
            
    def stop(self):
        """Stop desktop vision monitoring"""
        self.running = False
        cv2.destroyAllWindows()
    
    def set_gui_callback(self, callback):
        """Set callback for GUI updates (callback receives: frame, analysis, process_time)"""
        self.processor._gui_callback = callback