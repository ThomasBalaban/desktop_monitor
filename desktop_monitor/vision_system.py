"""
Desktop Vision Monitoring System

Captures and analyzes desktop visual changes using computer vision and LLM analysis
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

class VisionProcessor:
    """Processes visual frames for analysis"""
    
    def __init__(self, config):
        self.config = config
        self.analysis_history = deque(maxlen=10)
        self.summarized_context = deque(maxlen=3)
        self.summary_lock = threading.Lock()
        self.last_summary_time = time.time()
        self.last_valid_frame = None
        
        # Initialize Ollama
        try:
            import ollama # type: ignore
            self.ollama = ollama
            # Warmup the model
            print(f"[VISION] Loading model: {config.VISION_MODEL}")
            self.ollama.generate(model=config.VISION_MODEL, prompt="Ready")
            print(f"[VISION] Model loaded successfully")
        except Exception as e:
            print(f"[VISION] Error loading Ollama model: {e}")
            raise
            
        # Analysis prompt
        self.prompt = """
        Analyze ONLY what is currently visible. Describe concisely in 1-3 sentences. Do not assume what or try to guess what the content is. Try to describe the subject focus, what they are, what they look like, what they are doing, and where they are.
        """
        
    def optimize_frame(self, frame):
        """Convert a PIL image to base64 encoded JPEG for LLM processing"""
        frame = frame.resize(self.config.FRAME_RESIZE, Image.BILINEAR)
        buffered = BytesIO()
        frame.save(buffered, format="JPEG", quality=95, optimize=True)
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
        
        if change_percent > self.config.MIN_FRAME_CHANGE:
            print(f"[VISION] Screen changed ({change_percent:.2f})")
            return True
        return False
    
    def analyze_frame(self, frame):
        """Analyze a single frame using the LLM"""
        start_time = time.time()
        try:
            if not isinstance(frame, Image.Image):
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = frame
                
            response = self.ollama.chat(
                model=self.config.VISION_MODEL,
                messages=[{
                    "role": "user", 
                    "content": self.prompt, 
                    "images": [self.optimize_frame(pil_frame)]
                }],
                options={
                    'temperature': 0.2,
                    'num_ctx': 1024,
                    'num_gqa': 4,
                    'seed': int(time.time())
                }
            )
            result = response['message']['content'].strip()
            process_time = time.time() - start_time
            
            with self.summary_lock:
                self.analysis_history.append(result)
            
            # Display analysis with timing
            confidence = min(0.95, max(0.5, 1.0 - (process_time / 10.0)))
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp} | {process_time:.1f}s] [VISION ANALYSIS {confidence:.2f}] {result}")
            
            return result, process_time
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[VISION] Analysis error: {error_msg}")
            return error_msg, 0
    
    def generate_summary(self):
        """Generate a summary of recent visual activity"""
        with self.summary_lock:
            recent_analyses = list(self.analysis_history)[-5:]

        if not recent_analyses:
            return "No recent activity"

        summary_prompt = f"""Current situation summary from these events:
        {chr(10).join(recent_analyses)}
        Concise 1-3 sentence overview that attempts to guess what is happening currently from all of the image descriptions provided. Then try to guess what is currently happening. Try to follow the details of specific characters described."""
        
        try:
            response = self.ollama.generate(
                model=self.config.SUMMARY_MODEL,
                prompt=summary_prompt,
                options={'temperature': 0.2, 'num_predict': 250}
            )
            summary_text = response['response']
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [VISION SUMMARY] {summary_text}")
            
            with self.summary_lock:
                self.summarized_context.append(summary_text)
            
            return summary_text
        except Exception as e:
            error_msg = f"Summary error: {str(e)}"
            print(f"[VISION] {error_msg}")
            return error_msg
    
    def summary_worker(self):
        """Background worker to generate periodic summaries"""
        while True:
            time.sleep(1)
            if time.time() - self.last_summary_time > self.config.SUMMARY_INTERVAL:
                self.generate_summary()
                self.last_summary_time = time.time()

class DesktopVisionMonitor:
    """Main desktop vision monitoring class"""
    
    def __init__(self, config):
        self.config = config
        self.processor = VisionProcessor(config)
        self.running = False
        
        # Set up OpenCV
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
        
    def start(self):
        """Start desktop vision monitoring"""
        print(f"[VISION] Model: {self.config.VISION_MODEL}")
        print(f"[VISION] Monitor area: {self.config.MONITOR_AREA}")
        print(f"[VISION] Change threshold: {self.config.MIN_FRAME_CHANGE}")
        
        self.running = True
        
        # Start summary worker thread
        summary_thread = threading.Thread(target=self.processor.summary_worker, daemon=True)
        summary_thread.start()
        
        print("[VISION] Vision system initialized")
        
        try:
            with mss() as sct:
                print("[VISION] Monitoring desktop visual changes...")
                
                while self.running:
                    try:
                        # Capture frame
                        sct_img = sct.grab(self.config.MONITOR_AREA)
                        
                        # Create PIL Image from screenshot
                        frame = Image.frombytes(
                            'RGB', 
                            (sct_img.width, sct_img.height), 
                            sct_img.rgb
                        )
                        
                        # Check if frame changed enough to process
                        if self.processor.validate_frame(frame):
                            self.processor.last_valid_frame = frame.copy()
                            
                            # Analyze frame
                            self.processor.analyze_frame(self.processor.last_valid_frame)
                        else:
                            # Small sleep to prevent CPU overload
                            time.sleep(0.1)

                        # Exit check
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                            
                    except Exception as e:
                        print(f"[VISION] Frame capture error: {str(e)}")
                        time.sleep(1)
                        
        except Exception as e:
            print(f"[VISION] Error in vision monitoring: {e}")
        finally:
            cv2.destroyAllWindows()
            self.running = False
            
    def stop(self):
        """Stop desktop vision monitoring"""
        print("[VISION] Stopping desktop vision monitoring...")
        self.running = False
        cv2.destroyAllWindows()