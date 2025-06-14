"""
Enhanced Desktop Vision Monitoring System - Visual Narrator Mode

Updated to provide detailed visual descriptions for external AI consumption.
Focus on rich character, environment, and action descriptions.
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
    """Enhanced processor focused on detailed visual narration"""
    
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
            print(f"[VISION] Loading visual narrator model: {config.VISION_MODEL}")
            self.ollama.generate(model=config.VISION_MODEL, prompt="Ready for visual narration")
            print(f"[VISION] Visual narrator model loaded successfully")
        except Exception as e:
            print(f"[VISION] Error loading Ollama model: {e}")
            raise
            
        # Enhanced analysis prompt for detailed visual description
        self.prompt = """
Provide a detailed visual description of what's currently on screen. Describe:

1. CHARACTERS/PEOPLE: Appearance, clothing, position, current actions
2. ENVIRONMENT/SETTING: Location, colors, objects, layout, atmosphere  
3. ACTIVITIES: What's happening right now, movements, interactions
4. NOTABLE OBJECTS: Important items, tools, vehicles, or visual elements

Focus on observable details. Don't identify specific games, shows, or content - just describe what you see. Be specific about positions (left/right/center), colors, and actions. Provide enough detail that someone could understand the scene without seeing it.
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
            print(f"[VISION] Screen changed ({change_percent:.2f}) - generating new description")
            return True
        return False
    
    def analyze_frame(self, frame):
        """Generate detailed visual description using the LLM"""
        start_time = time.time()
        try:
            if not isinstance(frame, Image.Image):
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = frame
                
            # Use enhanced settings for detailed descriptions
            temperature = getattr(self.config, 'VISUAL_DESCRIPTION_TEMPERATURE', 0.1)
            context_size = getattr(self.config, 'VISUAL_DESCRIPTION_CONTEXT', 1300)
                
            response = self.ollama.chat(
                model=self.config.VISION_MODEL,
                messages=[{
                    "role": "user", 
                    "content": self.prompt, 
                    "images": [self.optimize_frame(pil_frame)]
                }],
                options={
                    'temperature': temperature,
                    'num_ctx': context_size,
                    'num_gqa': 4,
                    'seed': int(time.time())
                }
            )
            result = response['message']['content'].strip()
            process_time = time.time() - start_time
            
            with self.summary_lock:
                self.analysis_history.append(result)
            
            # Display enhanced analysis
            confidence = min(0.95, max(0.6, 1.0 - (process_time / 15.0)))
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp} | {process_time:.1f}s] [VISUAL DESCRIPTION {confidence:.2f}] {result}")
            
            return result, process_time
        
        except Exception as e:
            error_msg = f"Visual description error: {str(e)}"
            print(f"[VISION] Analysis error: {error_msg}")
            return error_msg, 0
    
    def generate_summary(self):
        """Generate a summary of recent visual activity"""
        with self.summary_lock:
            recent_analyses = list(self.analysis_history)[-5:]

        if not recent_analyses:
            return "No recent activity"

        summary_prompt = f"""Based on these recent visual descriptions, provide a brief contextual summary of what's happening:

{chr(10).join(recent_analyses)}

Provide a 1-2 sentence summary that captures the overall activity or narrative flow without identifying specific content."""
        
        try:
            response = self.ollama.generate(
                model=self.config.SUMMARY_MODEL,
                prompt=summary_prompt,
                options={'temperature': 0.2, 'num_predict': 150}
            )
            summary_text = response['response']
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [VISUAL SUMMARY] {summary_text}")
            
            with self.summary_lock:
                self.summarized_context.append(summary_text)
            
            return summary_text
        except Exception as e:
            error_msg = f"Summary error: {str(e)}"
            print(f"[VISION] {error_msg}")
            return error_msg
    
    def generate_enhanced_summary(self, recent_audio=None, recent_objects=None):
        """Generate comprehensive summary combining visual + audio + objects"""
        with self.summary_lock:
            recent_analyses = list(self.analysis_history)[-3:]

        if not recent_analyses:
            return "No visual activity detected", None

        # Build comprehensive summary
        summary_parts = []
        
        # Get component weights (with defaults)
        separator = " || "
        
        # 1. PRIMARY VISUAL DESCRIPTION (highest priority)
        if recent_analyses:
            primary_visual = recent_analyses[-1]
            summary_parts.append(f"VISUAL: {primary_visual}")
        
        # 2. AUDIO CONTEXT (secondary priority)
        if recent_audio:
            if isinstance(recent_audio, list):
                audio_summary = " | ".join(recent_audio[-3:])
            else:
                audio_summary = str(recent_audio)
            summary_parts.append(f"AUDIO: {audio_summary}")
        
        # 3. OBJECT DETECTION (minimal priority)
        if recent_objects and recent_objects != "No objects detected":
            summary_parts.append(f"OBJECTS: {recent_objects}")
        
        # Combine all parts
        comprehensive_summary = separator.join(summary_parts)
        
        # Generate meta-summary for context if multiple descriptions available
        meta_summary = None
        if len(recent_analyses) > 1:
            try:
                context_prompt = f"""Based on these recent visual descriptions, provide a brief contextual summary:

{chr(10).join(recent_analyses)}

Provide 1-2 sentences capturing the overall activity or narrative flow."""
                
                response = self.ollama.generate(
                    model=self.config.SUMMARY_MODEL,
                    prompt=context_prompt,
                    options={'temperature': 0.2, 'num_predict': 100}
                )
                meta_summary = response['response'].strip()
                
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] [VISUAL CONTEXT] {meta_summary}")
                
                with self.summary_lock:
                    self.summarized_context.append(meta_summary)
                
            except Exception as e:
                print(f"[VISION] Meta-summary error: {str(e)}")
        
        return comprehensive_summary, meta_summary
    
    def summary_worker(self):
        """Background worker to generate periodic summaries"""
        while True:
            time.sleep(1)
            if time.time() - self.last_summary_time > self.config.SUMMARY_INTERVAL:
                self.generate_summary()
                self.last_summary_time = time.time()

class DesktopVisionMonitor:
    """Enhanced desktop vision monitoring class focused on visual narration"""
    
    def __init__(self, config):
        self.config = config
        self.processor = VisionProcessor(config)
        self.running = False
        
        # Set up OpenCV
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
        
    def start(self):
        """Start enhanced desktop vision monitoring"""
        print(f"[VISION] Enhanced Visual Descriptions Enabled")
        print(f"[VISION] Model: {self.config.VISION_MODEL}")
        print(f"[VISION] Monitor area: {self.config.MONITOR_AREA}")
        print(f"[VISION] Change threshold: {self.config.MIN_FRAME_CHANGE}")
        print(f"[VISION] Focus: Detailed visual descriptions for external AI consumption")
        
        self.running = True
        
        # Start summary worker thread
        summary_thread = threading.Thread(target=self.processor.summary_worker, daemon=True)
        summary_thread.start()
        
        print("[VISION] Enhanced vision system initialized")
        
        try:
            with mss() as sct:
                print("[VISION] Monitoring for detailed visual descriptions...")
                
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
                            
                            # Generate detailed visual description
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
            print(f"[VISION] Error in enhanced vision monitoring: {e}")
        finally:
            cv2.destroyAllWindows()
            self.running = False
            
    def stop(self):
        """Stop enhanced desktop vision monitoring"""
        print("[VISION] Stopping enhanced visual narrator...")
        self.running = False
        cv2.destroyAllWindows()
    
    def get_latest_description(self):
        """Get the most recent visual description"""
        if self.processor.analysis_history:
            return self.processor.analysis_history[-1]
        return "No visual description available"
    
    def get_comprehensive_summary(self, audio_context=None, object_context=None):
        """Get comprehensive summary including visual, audio, and object context"""
        return self.processor.generate_enhanced_summary(audio_context, object_context)