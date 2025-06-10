"""
YOLO-based Desktop Vision Monitoring System with Enhanced Summaries

Real-time object detection for desktop monitoring using YOLO models.
Now includes advanced spatial, temporal, and contextual analysis.
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
    """Represents a single YOLO detection with enhanced properties"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    timestamp: float
    
    # Enhanced properties (will be calculated)
    relative_size: float = 0.0  # Percentage of screen
    screen_region: str = ""     # "top-left", "center", etc.
    confidence_level: str = ""  # "high", "medium", "low"

class EnhancedObjectTracker:
    """Advanced object tracking with spatial, temporal, and contextual analysis"""
    
    def __init__(self, max_history: int = 50, screen_width: int = 1920, screen_height: int = 1080):
        self.max_history = max_history
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_area = screen_width * screen_height
        
        # Enhanced tracking data
        self.object_history = deque(maxlen=max_history)
        self.class_counts = defaultdict(int)
        self.spatial_history = defaultdict(list)
        self.confidence_history = defaultdict(list)
        self.last_significant_change = time.time()
        
        # Scene analysis
        self.scene_type_history = deque(maxlen=10)
        self.stability_scores = deque(maxlen=20)
        self.object_relationships = {}
        
        # Summary preferences (configurable)
        self.summary_style = "scene"  # Options: basic, spatial, confidence, scene, prominence, temporal
        
    def set_screen_dimensions(self, width: int, height: int):
        """Update screen dimensions for spatial analysis"""
        self.screen_width = width
        self.screen_height = height
        self.screen_area = width * height
        
    def _get_screen_region(self, center: Tuple[int, int]) -> str:
        """Determine which region of screen the object is in"""
        x, y = center
        
        # Normalize coordinates
        x_norm = x / max(self.screen_width, 1)
        y_norm = y / max(self.screen_height, 1)
        
        # Define regions
        if x_norm < 0.33:
            x_region = "left"
        elif x_norm < 0.67:
            x_region = "center"
        else:
            x_region = "right"
            
        if y_norm < 0.33:
            y_region = "top"
        elif y_norm < 0.67:
            y_region = "middle"
        else:
            y_region = "bottom"
            
        if x_region == "center" and y_region == "middle":
            return "center"
        else:
            return f"{y_region}-{x_region}"
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence levels"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _detect_object_relationships(self, detections: List[Detection]) -> Dict[str, List[str]]:
        """Detect spatial relationships between objects"""
        relationships = defaultdict(list)
        
        # Define object categories
        workspace_objects = ['laptop', 'mouse', 'keyboard', 'monitor', 'computer']
        mobile_objects = ['cell phone', 'tablet']
        entertainment_objects = ['tv', 'remote']
        
        # Group objects by type
        workspace_items = [d for d in detections if d.class_name in workspace_objects]
        mobile_items = [d for d in detections if d.class_name in mobile_objects]
        entertainment_items = [d for d in detections if d.class_name in entertainment_objects]
        people = [d for d in detections if d.class_name == 'person']
        
        # Detect workspace clusters
        if len(workspace_items) >= 2:
            workspace_centers = [item.center for item in workspace_items]
            if self._objects_clustered(workspace_centers):
                relationships["workspace_cluster"] = [item.class_name for item in workspace_items]
        
        # Detect person-object interactions
        for i, person in enumerate(people):
            nearby_objects = []
            for detection in detections:
                if detection.class_name != 'person':
                    distance = self._calculate_distance(person.center, detection.center)
                    # Adjust interaction distance based on screen size
                    interaction_threshold = min(300, max(self.screen_width, self.screen_height) * 0.15)
                    if distance < interaction_threshold:
                        nearby_objects.append(detection.class_name)
            
            if nearby_objects:
                relationships[f"person_{i+1}_using"] = nearby_objects
        
        # Detect entertainment setup
        if entertainment_items:
            relationships["entertainment_setup"] = [item.class_name for item in entertainment_items]
        
        return dict(relationships)
    
    def _objects_clustered(self, centers: List[Tuple[int, int]], threshold_ratio: float = 0.2) -> bool:
        """Check if objects are spatially clustered"""
        if len(centers) < 2:
            return False
            
        # Calculate threshold based on screen size
        threshold = min(self.screen_width, self.screen_height) * threshold_ratio
        
        # Calculate average distance between all objects
        total_distance = 0
        pairs = 0
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = self._calculate_distance(centers[i], centers[j])
                total_distance += distance
                pairs += 1
        
        avg_distance = total_distance / pairs if pairs > 0 else float('inf')
        return avg_distance < threshold
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _classify_scene(self, detections: List[Detection], relationships: Dict) -> str:
        """Classify the type of scene based on objects and relationships"""
        object_names = [d.class_name for d in detections]
        person_count = object_names.count('person')
        
        # Workspace detection
        workspace_objects = ['laptop', 'computer', 'keyboard', 'mouse', 'monitor']
        if any(obj in object_names for obj in workspace_objects):
            if person_count >= 1:
                if 'workspace_cluster' in relationships:
                    return f"active_workspace_{person_count}p"
                else:
                    return f"casual_workspace_{person_count}p"
            else:
                return "unattended_workspace"
        
        # Entertainment detection
        if 'tv' in object_names or 'remote' in object_names:
            if person_count >= 1:
                return f"entertainment_{person_count}p"
            else:
                return "entertainment_idle"
        
        # Mobile/casual detection
        mobile_objects = ['cell phone', 'tablet']
        if any(obj in object_names for obj in mobile_objects) and person_count >= 1:
            return f"mobile_usage_{person_count}p"
        
        # Social/meeting detection
        if person_count >= 3:
            return "group_meeting"
        elif person_count == 2:
            return "collaboration"
        elif person_count == 1:
            return "individual_activity"
        else:
            return "empty_scene"
    
    def _detect_activity_patterns(self, enhanced_detections: List[Detection]) -> List[str]:
        """Analyze movement patterns across frames to detect activities"""
        activities = []
        
        # Need at least 5 frames for pattern analysis
        if len(self.object_history) < 5:
            return activities
        
        # Get recent frames (last 10 frames = ~5 seconds at 2 FPS)
        recent_frames = list(self.object_history)[-10:]
        current_time = time.time()
        
        # Analyze person activities
        person_activities = self._analyze_person_activities(recent_frames, enhanced_detections)
        activities.extend(person_activities)
        
        # Analyze object usage patterns
        object_activities = self._analyze_object_usage_patterns(recent_frames, enhanced_detections)
        activities.extend(object_activities)
        
        # Analyze scene dynamics
        scene_activities = self._analyze_scene_dynamics(recent_frames)
        activities.extend(scene_activities)
        
        return activities
    
    def _analyze_person_activities(self, recent_frames: List[Dict], current_detections: List[Detection]) -> List[str]:
        """Analyze person movement and positioning to infer activities"""
        activities = []
        
        # Get current people
        current_people = [d for d in current_detections if d.class_name == 'person']
        if not current_people:
            return activities
        
        # Track each person's movement
        for person_idx, current_person in enumerate(current_people):
            person_id = f"person_{person_idx + 1}"
            
            # Get historical positions for this person (simple tracking by position similarity)
            person_positions = []
            person_areas = []
            
            for frame in recent_frames:
                frame_people = [d for d in frame.get('enhanced_detections', []) if d.class_name == 'person']
                
                # Find closest person in this frame (simple tracking)
                if frame_people:
                    closest_person = min(frame_people, 
                                       key=lambda p: self._calculate_distance(current_person.center, p.center))
                    
                    # Only consider if reasonably close (same person)
                    if self._calculate_distance(current_person.center, closest_person.center) < 200:
                        person_positions.append(closest_person.center)
                        person_areas.append(closest_person.area)
            
            if len(person_positions) < 3:
                continue
                
            # Analyze movement patterns
            activity = self._classify_person_movement(person_positions, person_areas, current_person)
            if activity:
                activities.append(f"{person_id} {activity}")
        
        return activities
    
    def _classify_person_movement(self, positions: List[Tuple[int, int]], areas: List[int], current_person: Detection) -> str:
        """Classify person's activity based on movement patterns"""
        if len(positions) < 3:
            return ""
        
        # Calculate movement metrics
        total_movement = 0
        rapid_movements = 0
        position_changes = []
        
        for i in range(1, len(positions)):
            distance = self._calculate_distance(positions[i-1], positions[i])
            total_movement += distance
            position_changes.append(distance)
            
            if distance > 50:  # Rapid movement threshold
                rapid_movements += 1
        
        avg_movement = total_movement / len(position_changes) if position_changes else 0
        movement_variance = np.var(position_changes) if len(position_changes) > 1 else 0
        
        # Calculate area changes (size variation might indicate moving closer/farther)
        area_changes = []
        if len(areas) > 1:
            for i in range(1, len(areas)):
                area_change = abs(areas[i] - areas[i-1]) / max(areas[i-1], 1)
                area_changes.append(area_change)
        
        avg_area_change = np.mean(area_changes) if area_changes else 0
        
        # Analyze current position context
        workspace_nearby = self._person_near_workspace(current_person)
        entertainment_nearby = self._person_near_entertainment(current_person)
        
        # Activity classification logic
        if avg_movement > 100:  # High movement
            if movement_variance > 2000:  # Erratic movement
                return "dancing/exercising"
            else:
                return "walking/pacing"
        elif avg_movement > 30:  # Medium movement
            if workspace_nearby:
                return "working actively"
            elif rapid_movements >= 3:
                return "gesturing/presenting"
            else:
                return "moving around"
        elif avg_movement < 10:  # Low movement
            if avg_area_change > 0.1:  # Size changing but not position
                return "moving in place"
            elif workspace_nearby:
                return "sitting/working"
            elif entertainment_nearby:
                return "watching/relaxing"
            else:
                return "standing still"
        else:  # Medium-low movement
            if workspace_nearby:
                return "working/typing"
            else:
                return "casual movement"
    
    def _person_near_workspace(self, person: Detection) -> bool:
        """Check if person is near workspace objects"""
        if not self.object_history:
            return False
        
        latest_frame = self.object_history[-1]
        workspace_objects = ['laptop', 'keyboard', 'mouse', 'monitor', 'computer']
        
        for detection in latest_frame.get('enhanced_detections', []):
            if detection.class_name in workspace_objects:
                distance = self._calculate_distance(person.center, detection.center)
                if distance < 300:  # Within workspace range
                    return True
        return False
    
    def _person_near_entertainment(self, person: Detection) -> bool:
        """Check if person is near entertainment objects"""
        if not self.object_history:
            return False
        
        latest_frame = self.object_history[-1]
        entertainment_objects = ['tv', 'remote', 'couch']
        
        for detection in latest_frame.get('enhanced_detections', []):
            if detection.class_name in entertainment_objects:
                distance = self._calculate_distance(person.center, detection.center)
                if distance < 400:  # Within entertainment range
                    return True
        return False
    
    def _analyze_object_usage_patterns(self, recent_frames: List[Dict], current_detections: List[Detection]) -> List[str]:
        """Analyze how objects are being used based on patterns"""
        activities = []
        
        # Look for objects that appear/disappear frequently (active usage)
        object_appearances = defaultdict(int)
        object_disappearances = defaultdict(int)
        
        for i, frame in enumerate(recent_frames[1:], 1):
            prev_frame = recent_frames[i-1]
            current_objects = set(frame.get('class_counts', {}).keys())
            prev_objects = set(prev_frame.get('class_counts', {}).keys())
            
            # Objects that appeared
            for obj in current_objects - prev_objects:
                object_appearances[obj] += 1
            
            # Objects that disappeared
            for obj in prev_objects - current_objects:
                object_disappearances[obj] += 1
        
        # Detect active usage patterns
        usage_objects = ['cell phone', 'remote', 'book', 'cup', 'bottle']
        for obj in usage_objects:
            if object_appearances[obj] + object_disappearances[obj] >= 3:
                activities.append(f"actively using {obj}")
        
        # Look for new objects that suggest activities
        if len(recent_frames) >= 2:
            current_objects = set(d.class_name for d in current_detections)
            prev_objects = set(recent_frames[-2].get('class_counts', {}).keys())
            new_objects = current_objects - prev_objects
            
            for obj in new_objects:
                if obj == 'book':
                    activities.append("started reading")
                elif obj == 'cell phone':
                    activities.append("picked up phone")
                elif obj in ['cup', 'bottle']:
                    activities.append("drinking/eating")
        
        return activities
    
    def _analyze_scene_dynamics(self, recent_frames: List[Dict]) -> List[str]:
        """Analyze overall scene changes and dynamics"""
        activities = []
        
        if len(recent_frames) < 5:
            return activities
        
        # Analyze stability trends
        stability_scores = [frame.get('stability', 1.0) for frame in recent_frames]
        recent_stability = np.mean(stability_scores[-3:])  # Last 3 frames
        
        # Analyze scene type changes
        scene_types = [frame.get('scene_type', 'unknown') for frame in recent_frames]
        unique_recent_scenes = set(scene_types[-5:])
        
        # Detect scene transitions
        if len(unique_recent_scenes) > 2:
            activities.append("scene changing")
        elif recent_stability < 0.5:
            activities.append("high activity")
        elif recent_stability > 0.9:
            activities.append("stable scene")
        
        # Detect group dynamics
        person_counts = []
        for frame in recent_frames:
            counts = frame.get('class_counts', {})
            person_counts.append(counts.get('person', 0))
        
        if len(set(person_counts)) > 1:  # Person count changing
            max_people = max(person_counts)
            min_people = min(person_counts)
            if max_people > min_people:
                if max_people >= 2:
                    activities.append("people joining/leaving")
                else:
                    activities.append("person entering/leaving")
        
        return activities
    
    def _calculate_scene_stability(self, current_objects: Dict[str, int]) -> float:
        """Calculate how stable the scene is (0.0 = very unstable, 1.0 = very stable)"""
        if len(self.object_history) < 2:
            return 1.0
        
        # Compare with recent history
        recent_frames = list(self.object_history)[-5:]
        
        stability_scores = []
        for frame in recent_frames:
            frame_objects = frame.get('class_counts', {})
            
            # Calculate similarity
            all_classes = set(current_objects.keys()) | set(frame_objects.keys())
            differences = 0
            
            for class_name in all_classes:
                current_count = current_objects.get(class_name, 0)
                frame_count = frame_objects.get(class_name, 0)
                differences += abs(current_count - frame_count)
            
            # Normalize by total objects
            total_objects = sum(current_objects.values()) + sum(frame_objects.values())
            if total_objects > 0:
                similarity = 1.0 - (differences / total_objects)
                stability_scores.append(max(0.0, similarity))
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _enhance_detections(self, detections: List[Detection]) -> List[Detection]:
        """Add enhanced properties to detections"""
        enhanced_detections = []
        
        for det in detections:
            # Calculate enhanced properties
            det.relative_size = (det.area / max(self.screen_area, 1)) * 100
            det.screen_region = self._get_screen_region(det.center)
            det.confidence_level = self._get_confidence_level(det.confidence)
            enhanced_detections.append(det)
        
        return enhanced_detections
    
    def update(self, detections: List[Detection]) -> Dict:
        """Update tracker with new detections and return comprehensive analysis"""
        current_time = time.time()
        
        # Enhance detections with spatial and confidence info
        enhanced_detections = self._enhance_detections(detections)
        
        # Update class counts
        new_counts = defaultdict(int)
        for detection in enhanced_detections:
            new_counts[detection.class_name] += 1
            
        # Detect changes
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
        
        # Detect relationships and classify scene
        relationships = self._detect_object_relationships(enhanced_detections)
        scene_type = self._classify_scene(enhanced_detections, relationships)
        stability = self._calculate_scene_stability(dict(new_counts))
        
        # NEW: Detect activity patterns
        activities = self._detect_activity_patterns(enhanced_detections)
        
        # Update history
        self.object_history.append({
            'timestamp': current_time,
            'detections': enhanced_detections,
            'class_counts': dict(new_counts),
            'changes': changes,
            'relationships': relationships,
            'scene_type': scene_type,
            'stability': stability,
            'activities': activities  # NEW: Store detected activities
        })
        
        self.class_counts = new_counts
        
        if changes:
            self.last_significant_change = current_time
            
        return {
            'total_objects': len(enhanced_detections),
            'class_counts': dict(new_counts),
            'changes': changes,
            'time_since_change': current_time - self.last_significant_change,
            'relationships': relationships,
            'scene_type': scene_type,
            'stability': stability,
            'enhanced_detections': enhanced_detections,
            'activities': activities  # NEW: Return detected activities
        }
    
    def get_summary(self, style: Optional[str] = None) -> str:
        """Get enhanced summary based on specified style"""
        if not self.class_counts:
            return "No objects detected"
        
        # Use specified style or default
        summary_style = style or self.summary_style
        
        # Get latest analysis data
        latest_frame = self.object_history[-1] if self.object_history else {}
        enhanced_detections = latest_frame.get('enhanced_detections', [])
        relationships = latest_frame.get('relationships', {})
        scene_type = latest_frame.get('scene_type', 'unknown')
        stability = latest_frame.get('stability', 1.0)
        
        if summary_style == "basic":
            return self._get_basic_summary()
        elif summary_style == "spatial":
            return self._get_spatial_summary(enhanced_detections)
        elif summary_style == "confidence":
            return self._get_confidence_summary(enhanced_detections)
        elif summary_style == "scene":
            return self._get_scene_summary(scene_type, relationships, stability)
        elif summary_style == "prominence":
            return self._get_prominence_summary(enhanced_detections)
        elif summary_style == "temporal":
            return self._get_temporal_summary()
        else:
            # Default to scene summary
            return self._get_scene_summary(scene_type, relationships, stability)
    
    def _get_basic_summary(self) -> str:
        """Basic object count summary"""
        summary_parts = []
        for class_name, count in sorted(self.class_counts.items()):
            if count == 1:
                summary_parts.append(class_name)
            else:
                summary_parts.append(f"{count} {class_name}s")
        return ", ".join(summary_parts)
    
    def _get_spatial_summary(self, enhanced_detections: List[Detection]) -> str:
        """Spatial location-aware summary"""
        if not enhanced_detections:
            return "No objects detected"
        
        spatial_parts = []
        for det in enhanced_detections:
            spatial_parts.append(f"{det.class_name} ({det.screen_region})")
        return ", ".join(spatial_parts)
    
    def _get_confidence_summary(self, enhanced_detections: List[Detection]) -> str:
        """Confidence-aware summary"""
        if not enhanced_detections:
            return "No objects detected"
        
        conf_parts = []
        for det in enhanced_detections:
            conf_parts.append(f"{det.class_name} ({det.confidence:.2f}-{det.confidence_level})")
        return ", ".join(conf_parts)
    
    def _get_scene_summary(self, scene_type: str, relationships: Dict, stability: float) -> str:
        """Contextual scene analysis summary with activities"""
        scene_parts = [f"Scene: {scene_type}"]
        
        # Add activity information if available
        if self.object_history:
            latest_frame = self.object_history[-1]
            activities = latest_frame.get('activities', [])
            if activities:
                scene_parts.append(f"Activities: {', '.join(activities)}")
        
        if relationships:
            rel_descriptions = []
            for rel_type, objects in relationships.items():
                if "workspace_cluster" in rel_type:
                    rel_descriptions.append(f"workspace: {', '.join(objects)}")
                elif "person_" in rel_type and "_using" in rel_type:
                    person_num = rel_type.split('_')[1]
                    rel_descriptions.append(f"person{person_num} using: {', '.join(objects)}")
                elif "entertainment_setup" in rel_type:
                    rel_descriptions.append(f"entertainment: {', '.join(objects)}")
            
            if rel_descriptions:
                scene_parts.extend(rel_descriptions)
        
        scene_parts.append(f"stability: {stability:.0%}")
        return " | ".join(scene_parts)
    
    def _get_prominence_summary(self, enhanced_detections: List[Detection]) -> str:
        """Size and visual prominence summary"""
        if not enhanced_detections:
            return "No prominent objects"
        
        # Sort by size
        sorted_by_size = sorted(enhanced_detections, key=lambda x: x.relative_size, reverse=True)
        size_parts = []
        
        for det in sorted_by_size[:3]:  # Top 3 largest
            if det.relative_size > 5:
                size_desc = "large"
            elif det.relative_size > 1:
                size_desc = "medium"
            else:
                size_desc = "small"
            size_parts.append(f"{det.class_name} ({size_desc}-{det.relative_size:.1f}%)")
        
        return f"Dominated by: {', '.join(size_parts)}"
    
    def _get_temporal_summary(self) -> str:
        """Temporal change-aware summary with activity detection"""
        basic_summary = self._get_basic_summary()
        
        if len(self.object_history) >= 2:
            latest_frame = self.object_history[-1]
            changes = latest_frame.get('changes', [])
            stability = latest_frame.get('stability', 1.0)
            activities = latest_frame.get('activities', [])
            
            change_summary = f"Current: {basic_summary}"
            if activities:
                change_summary += f" | Activities: {', '.join(activities)}"
            if changes:
                change_summary += f" | Changes: {', '.join(changes)}"
            change_summary += f" | Stability: {stability:.0%}"
            
            return change_summary
        else:
            return basic_summary
    
    def get_all_summaries(self) -> Dict[str, str]:
        """Get all available summary types"""
        return {
            'basic': self.get_summary('basic'),
            'spatial': self.get_summary('spatial'),
            'confidence': self.get_summary('confidence'),
            'scene': self.get_summary('scene'),
            'prominence': self.get_summary('prominence'),
            'temporal': self.get_summary('temporal')
        }

class YOLOProcessor:
    """Processes frames using YOLO object detection with enhanced analysis"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
        # Initialize enhanced tracker with screen dimensions
        screen_area = config.MONITOR_AREA
        self.tracker = EnhancedObjectTracker(
            screen_width=screen_area.get('width', 1920),
            screen_height=screen_area.get('height', 1080)
        )
        
        # Set summary style from config if available
        summary_style = getattr(config, 'YOLO_SUMMARY_STYLE', 'scene')
        self.tracker.summary_style = summary_style
        
        self.processing_lock = threading.Lock()
        self.last_detection_time = 0
        self.detection_interval = 1.0 / config.YOLO_FPS
        
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
        """Process frame and return enhanced detection results"""
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
            
            # Update enhanced tracker
            analysis = self.tracker.update(detections)
            
            # Create result with enhanced data
            result = {
                'detections': detections,
                'analysis': analysis,
                'processing_time': time.time() - current_time,
                'frame_timestamp': current_time,
                'enhanced_summary': self.tracker.get_summary(),
                'all_summaries': self.tracker.get_all_summaries()
            }
            
            return result
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection bounding boxes on frame with enhanced info"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Color based on confidence level
            if detection.confidence >= 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif detection.confidence >= 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw enhanced label with region info
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if hasattr(detection, 'screen_region') and detection.screen_region:
                label += f" ({detection.screen_region})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame

class DesktopYOLOMonitor:
    """Main YOLO-based desktop vision monitoring class with enhanced summaries"""
    
    def __init__(self, config):
        self.config = config
        self.processor = YOLOProcessor(config)
        self.running = False
        self.detection_callback = None
        
        # Statistics
        self.total_detections = 0
        self.total_frames_processed = 0
        self.start_time = None
        
        # Enhanced logging options
        self.verbose_summaries = getattr(config, 'YOLO_VERBOSE_SUMMARIES', False)
        self.summary_rotation = getattr(config, 'YOLO_SUMMARY_ROTATION', False)
        self.current_summary_style = 0
        self.summary_styles = ['basic', 'spatial', 'scene', 'temporal']
        
    def set_detection_callback(self, callback):
        """Set callback function to receive detection results"""
        self.detection_callback = callback
    
    def _log_detections(self, result: Dict):
        """Log enhanced detection results to console"""
        detections = result['detections']
        analysis = result['analysis']
        enhanced_summary = result['enhanced_summary']
        
        if detections or analysis['changes']:
            timestamp = time.strftime("%H:%M:%S")
            processing_time = result['processing_time']
            
            # Primary summary output
            if self.summary_rotation:
                # Rotate through different summary styles
                style = self.summary_styles[self.current_summary_style]
                summary = self.processor.tracker.get_summary(style)
                style_indicator = f"[{style.upper()}]"
                self.current_summary_style = (self.current_summary_style + 1) % len(self.summary_styles)
            else:
                summary = enhanced_summary
                style_indicator = ""
            
            print(f"[{timestamp} | {processing_time:.2f}s] [YOLO DETECTION] {style_indicator} {summary}")
            
            # Log changes if any
            if analysis['changes']:
                changes_str = ", ".join(analysis['changes'])
                print(f"[{timestamp}] [YOLO CHANGES] {changes_str}")
            
            # Verbose summaries - show all summary types
            if self.verbose_summaries:
                all_summaries = result['all_summaries']
                for summary_type, summary_text in all_summaries.items():
                    if summary_type != 'basic':  # Skip basic to avoid redundancy
                        print(f"[{timestamp}] [YOLO {summary_type.upper()}] {summary_text}")
            
            # Log activity detection
            analysis = result['analysis']
            activities = analysis.get('activities', [])
            if activities:
                activity_str = ", ".join(activities)
                print(f"[{timestamp}] [YOLO ACTIVITIES] {activity_str}")
            high_conf_detections = [d for d in detections if d.confidence > 0.8]
            if high_conf_detections:
                for detection in high_conf_detections[:3]:  # Limit to top 3
                    region_info = f" in {detection.screen_region}" if hasattr(detection, 'screen_region') else ""
                    print(f"[{timestamp}] [YOLO HIGH-CONF] {detection.class_name} ({detection.confidence:.2f}){region_info}")
    
    def start(self):
        """Start enhanced YOLO desktop monitoring"""
        print(f"[YOLO] Enhanced Model: {self.config.YOLO_MODEL}")
        print(f"[YOLO] Confidence threshold: {self.config.YOLO_CONFIDENCE}")
        print(f"[YOLO] Target FPS: {self.config.YOLO_FPS}")
        print(f"[YOLO] Monitor area: {self.config.MONITOR_AREA}")
        print(f"[YOLO] Summary style: {self.processor.tracker.summary_style}")
        if self.verbose_summaries:
            print(f"[YOLO] Verbose summaries: enabled")
        if self.summary_rotation:
            print(f"[YOLO] Summary rotation: enabled ({', '.join(self.summary_styles)})")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            with mss() as sct:
                print("[YOLO] Starting enhanced object detection...")
                
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
                            
                            # Log enhanced detections
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
            print(f"[YOLO] Error in enhanced YOLO monitoring: {e}")
        finally:
            self.running = False
            self._print_statistics()
    
    def _print_statistics(self):
        """Print enhanced detection statistics"""
        if self.start_time:
            runtime = time.time() - self.start_time
            avg_fps = self.total_frames_processed / runtime if runtime > 0 else 0
            avg_detections_per_frame = self.total_detections / max(self.total_frames_processed, 1)
            
            print(f"[YOLO] Enhanced Statistics:")
            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Frames processed: {self.total_frames_processed}")
            print(f"  Total detections: {self.total_detections}")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Avg detections/frame: {avg_detections_per_frame:.1f}")
            
            # Scene type statistics
            if self.processor.tracker.object_history:
                scene_types = [frame.get('scene_type', 'unknown') for frame in self.processor.tracker.object_history]
                unique_scenes = list(set(scene_types))
                print(f"  Scene types observed: {', '.join(unique_scenes)}")
                
                # Stability statistics
                stability_scores = [frame.get('stability', 1.0) for frame in self.processor.tracker.object_history]
                avg_stability = np.mean(stability_scores) if stability_scores else 1.0
                print(f"  Average scene stability: {avg_stability:.1%}")
    
    def stop(self):
        """Stop enhanced YOLO desktop monitoring"""
        print("[YOLO] Stopping enhanced YOLO monitoring...")
        self.running = False
    
    def get_current_objects(self) -> str:
        """Get current enhanced object summary"""
        return self.processor.tracker.get_summary()
    
    def get_current_scene_analysis(self) -> Dict:
        """Get detailed current scene analysis"""
        if not self.processor.tracker.object_history:
            return {"scene_type": "empty", "relationships": {}, "stability": 1.0}
        
        latest_frame = self.processor.tracker.object_history[-1]
        return {
            "scene_type": latest_frame.get('scene_type', 'unknown'),
            "relationships": latest_frame.get('relationships', {}),
            "stability": latest_frame.get('stability', 1.0),
            "summary": self.processor.tracker.get_summary(),
            "all_summaries": self.processor.tracker.get_all_summaries()
        }
    
    def get_recent_changes(self, seconds: int = 30) -> List[str]:
        """Get recent object changes with enhanced context"""
        current_time = time.time()
        recent_changes = []
        
        for entry in self.processor.tracker.object_history:
            if current_time - entry['timestamp'] <= seconds:
                if entry['changes']:
                    timestamp = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                    changes_str = ", ".join(entry['changes'])
                    scene_type = entry.get('scene_type', 'unknown')
                    recent_changes.append(f"[{timestamp}] {changes_str} (scene: {scene_type})")
        
        return recent_changes
    
    def set_summary_style(self, style: str) -> bool:
        """Change the summary style dynamically"""
        valid_styles = ['basic', 'spatial', 'confidence', 'scene', 'prominence', 'temporal']
        if style in valid_styles:
            self.processor.tracker.summary_style = style
            print(f"[YOLO] Summary style changed to: {style}")
            return True
        else:
            print(f"[YOLO] Invalid summary style: {style}. Valid options: {', '.join(valid_styles)}")
            return False
    
    def toggle_verbose_summaries(self):
        """Toggle verbose summary output"""
        self.verbose_summaries = not self.verbose_summaries
        status = "enabled" if self.verbose_summaries else "disabled"
        print(f"[YOLO] Verbose summaries {status}")
    
    def toggle_summary_rotation(self):
        """Toggle summary style rotation"""
        self.summary_rotation = not self.summary_rotation
        status = "enabled" if self.summary_rotation else "disabled"
        print(f"[YOLO] Summary rotation {status}")
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        if not self.start_time:
            return {}
        
        runtime = time.time() - self.start_time
        
        return {
            "runtime_seconds": runtime,
            "frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "average_fps": self.total_frames_processed / runtime if runtime > 0 else 0,
            "detections_per_frame": self.total_detections / max(self.total_frames_processed, 1),
            "tracker_history_size": len(self.processor.tracker.object_history),
            "current_scene": self.get_current_scene_analysis()
        }