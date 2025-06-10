"""
YOLO-based Desktop Vision Monitoring System

Real-time object detection for desktop monitoring using YOLO models.
Works alongside the existing LLM vision system to provide structured object detection.
"""

import cv2 # type: ignore
import time
import threading
import numpy as np # type: ignore
from mss import mss # type: ignore
from PIL import Image # type: ignore
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Detection:
    """Represents a single YOLO detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    timestamp: float

class ObjectTracker:
    """Tracks objects across frames to detect changes and movements"""
    
    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self.object_history = deque(maxlen=max_history)
        self.class_counts = defaultdict(int)
        self.last_significant_change = time.time()
        
    def update(self, detections: List[Detection]) -> Dict:
        """Update tracker with new detections and return analysis"""
        current_time = time.time()
        
        # Update class counts
        new_counts = defaultdict(int)
        for detection in detections:
            new_counts[detection.class_name] += 1
            
        # Detect significant changes
        changes = []
        for class_name, count in new_counts.items():
            prev_count = self.class_counts.get(class_name, 0)
            if count != prev_count:
                if count > prev_count:
                    changes.append(f"+{count-prev_count} {class_name}")
                else:
                    changes.append(f"-{prev_count-count} {class_name}")
                    
        # Check for new classes
        for class_name in new_counts:
            if class_name not in self.class_counts:
                changes.append(f"NEW: {class_name}")
                
        # Check for disappeared classes
        for class_name in self.class_counts:
            if class_name not in new_counts:
                changes.append(f"GONE: {class_name}")
        
        # Update history
        self.object_history.append({
            'timestamp': current_time,
            'detections': detections.copy(),
            'class_counts': dict(new_counts),
            'changes': changes
        })
        
        self.class_counts = new_counts
        
        if changes:
            self.last_significant_change = current_time
            
        return {
            'total_objects': len(detections),
            'class_counts': dict(new_counts),
            'changes': changes,
            'time_since_change': current_time - self.last_significant_change
        }
    
    def get_summary(self) -> str:
        """Get a text summary of current objects"""
        if not self.class_counts:
            return "No objects detected"
            
        summary_parts = []
        for class_name, count in sorted(self.class_counts.items()):
            if count == 1:
                summary_parts.append(class_name)
            else:
                summary_parts.append(f"{count} {class_name}s")
                
        return ", ".join(summary_parts)

class YOLOProcessor:
    """Processes frames using YOLO object detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tracker = ObjectTracker()
        self.processing_lock = threading.Lock()
        self.last_detection_time = 0
        self.detection_interval = 1.0 / config.YOLO_FPS  # Convert FPS to interval
        
        # YOLO class names (COCO dataset)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            import ultralytics # type: ignore
            from ultralytics import YOLO # type: ignore
            
            print(f"[YOLO] Loading model: {self.config.YOLO_MODEL}")
            self.model = YOLO(self.config.YOLO_MODEL)
            
            # Move to GPU if available
            if self.config.DEVICE == "cuda":
                self.model.to('cuda')
                
            # Warmup
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            
            print(f"[YOLO] Model loaded successfully on {self.config.DEVICE}")
            
        except ImportError:
            print("[YOLO] Error: ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            print(f"[YOLO] Error loading model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection on frame"""
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.config.YOLO_CONFIDENCE)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if conf >= self.config.YOLO_CONFIDENCE:
                            x1, y1, x2, y2 = map(int, box)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            area = (x2 - x1) * (y2 - y1)
                            
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            detection = Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=float(conf),
                                bbox=(x1, y1, x2, y2),
                                center=(center_x, center_y),
                                area=area,
                                timestamp=time.time()
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process frame and return detection results"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time < self.detection_interval:
            return None
            
        with self.processing_lock:
            self.last_detection_time = current_time
            
            # Resize frame if needed
            if hasattr(self.config, 'YOLO_INPUT_SIZE'):
                height, width = frame.shape[:2]
                target_size = self.config.YOLO_INPUT_SIZE
                if max(height, width) > target_size:
                    scale = target_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
            
            # Run detection
            detections = self.detect_objects(frame)
            
            # Update tracker
            analysis = self.tracker.update(detections)
            
            # Create result
            result = {
                'detections': detections,
                'analysis': analysis,
                'processing_time': time.time() - current_time,
                'frame_timestamp': current_time
            }
            
            return result
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection bounding boxes on frame (for debugging)"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame

class DesktopYOLOMonitor:
    """Main YOLO-based desktop vision monitoring class"""
    
    def __init__(self, config):
        self.config = config
        self.processor = YOLOProcessor(config)
        self.running = False
        self.detection_callback = None
        
        # Statistics
        self.total_detections = 0
        self.total_frames_processed = 0
        self.start_time = None
        
    def set_detection_callback(self, callback):
        """Set callback function to receive detection results"""
        self.detection_callback = callback
    
    def _log_detections(self, result: Dict):
        """Log detection results to console"""
        detections = result['detections']
        analysis = result['analysis']
        
        if detections or analysis['changes']:
            timestamp = time.strftime("%H:%M:%S")
            processing_time = result['processing_time']
            
            # Log summary
            summary = self.processor.tracker.get_summary()
            print(f"[{timestamp} | {processing_time:.2f}s] [YOLO DETECTION] {summary}")
            
            # Log changes if any
            if analysis['changes']:
                changes_str = ", ".join(analysis['changes'])
                print(f"[{timestamp}] [YOLO CHANGES] {changes_str}")
            
            # Log high-confidence detections
            high_conf_detections = [d for d in detections if d.confidence > 0.8]
            if high_conf_detections:
                for detection in high_conf_detections[:3]:  # Limit to top 3
                    print(f"[{timestamp}] [YOLO HIGH-CONF] {detection.class_name} ({detection.confidence:.2f}) at {detection.center}")
    
    def start(self):
        """Start YOLO desktop monitoring"""
        print(f"[YOLO] Model: {self.config.YOLO_MODEL}")
        print(f"[YOLO] Confidence threshold: {self.config.YOLO_CONFIDENCE}")
        print(f"[YOLO] Target FPS: {self.config.YOLO_FPS}")
        print(f"[YOLO] Monitor area: {self.config.MONITOR_AREA}")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            with mss() as sct:
                print("[YOLO] Starting object detection...")
                
                while self.running:
                    try:
                        # Capture frame
                        sct_img = sct.grab(self.config.MONITOR_AREA)
                        
                        # Convert to numpy array
                        frame = np.array(sct_img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                        
                        # Process frame
                        result = self.processor.process_frame(frame)
                        
                        if result:
                            self.total_frames_processed += 1
                            self.total_detections += len(result['detections'])
                            
                            # Log detections
                            self._log_detections(result)
                            
                            # Call user callback if set
                            if self.detection_callback:
                                self.detection_callback(result)
                        
                        # Small sleep to prevent CPU overload
                        time.sleep(0.01)
                        
                    except Exception as e:
                        print(f"[YOLO] Frame processing error: {e}")
                        time.sleep(1)
                        
        except Exception as e:
            print(f"[YOLO] Error in YOLO monitoring: {e}")
        finally:
            self.running = False
            self._print_statistics()
    
    def _print_statistics(self):
        """Print detection statistics"""
        if self.start_time:
            runtime = time.time() - self.start_time
            avg_fps = self.total_frames_processed / runtime if runtime > 0 else 0
            avg_detections_per_frame = self.total_detections / max(self.total_frames_processed, 1)
            
            print(f"[YOLO] Statistics:")
            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Frames processed: {self.total_frames_processed}")
            print(f"  Total detections: {self.total_detections}")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Avg detections/frame: {avg_detections_per_frame:.1f}")
    
    def stop(self):
        """Stop YOLO desktop monitoring"""
        print("[YOLO] Stopping YOLO monitoring...")
        self.running = False
    
    def get_current_objects(self) -> str:
        """Get current object summary"""
        return self.processor.tracker.get_summary()
    
    def get_recent_changes(self, seconds: int = 30) -> List[str]:
        """Get recent object changes"""
        current_time = time.time()
        recent_changes = []
        
        for entry in self.processor.tracker.object_history:
            if current_time - entry['timestamp'] <= seconds:
                if entry['changes']:
                    timestamp = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                    changes_str = ", ".join(entry['changes'])
                    recent_changes.append(f"[{timestamp}] {changes_str}")
        
        return recent_changes