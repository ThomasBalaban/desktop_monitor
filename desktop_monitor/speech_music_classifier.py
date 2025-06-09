"""
Speech/Music Classifier - Standalone Module

A dedicated classifier for distinguishing between speech and music in audio streams.
This module has been separated for easier debugging and tuning.

Features:
- Multi-feature analysis (zero-crossing, spectral bands, centroid)
- History-based smoothing for stable classification
- Configurable thresholds and parameters
- Detailed logging for debugging
- Confidence scoring

Usage:
    classifier = SpeechMusicClassifier(debug=True)
    audio_type, confidence = classifier.classify(audio_chunk)
"""

import numpy as np
import time
from typing import Tuple, List, Optional

class SpeechMusicClassifier:
    """
    Enhanced classifier to detect speech vs music with detailed debugging capabilities
    """
    
    def __init__(self, 
                 max_history: int = 5,
                 speech_threshold: float = 0.6,
                 music_threshold: float = 0.4,
                 min_confidence: float = 0.2,
                 debug: bool = False,
                 sample_rate: int = 16000):
        """
        Initialize the classifier
        
        Args:
            max_history: Number of previous classifications to consider
            speech_threshold: Fraction of history needed to classify as speech
            music_threshold: Fraction of history needed to classify as music  
            min_confidence: Minimum confidence to change classification
            debug: Enable detailed debug output
            sample_rate: Audio sample rate (Hz)
        """
        # Current state
        self.current_type = "speech"  # Default to speech
        self.sample_rate = sample_rate
        
        # History tracking
        self.history = []
        self.max_history = max_history
        
        # Thresholds
        self.speech_threshold = speech_threshold
        self.music_threshold = music_threshold
        self.min_confidence = min_confidence
        
        # Debug settings
        self.debug = debug
        self.classification_count = 0
        
        # Feature weights (adjustable for tuning)
        self.weights = {
            'zero_crossing': 1.0,
            'speech_band_energy': 1.0,
            'high_freq_energy': 1.5,
            'spectral_centroid': 1.0,
            'spectral_flatness': 1.0,
            'energy_variance': 1.0
        }
        
        if self.debug:
            print(f"[CLASSIFIER] Initialized with:")
            print(f"  History size: {max_history}")
            print(f"  Speech threshold: {speech_threshold}")
            print(f"  Music threshold: {music_threshold}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Feature weights: {self.weights}")
    
    def _log_debug(self, message: str):
        """Log debug message if debug mode is enabled"""
        if self.debug:
            print(f"[CLASSIFIER] {message}")
    
    def _extract_features(self, audio_chunk: np.ndarray) -> dict:
        """
        Extract audio features for classification
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Normalize the audio
            audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-10)
            
            # 1. Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_chunk)) != 0)
            features['zero_crossing_rate'] = zero_crossings / len(audio_chunk)
            
            # 2. Frequency analysis
            spectrum = np.abs(np.fft.rfft(audio_chunk))
            freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
            
            # 3. Energy in different frequency bands
            bands = [0, 150, 300, 1000, 2000, 3000, 5000, 8000, 12000]
            band_energy = []
            
            for i in range(len(bands)-1):
                mask = (freqs >= bands[i]) & (freqs < bands[i+1])
                band_energy.append(np.sum(spectrum[mask]))
                
            # Normalize band energies
            total_energy = sum(band_energy) + 1e-10
            features['band_energy_ratio'] = [e/total_energy for e in band_energy]
            
            # Speech band energy (1000-3000 Hz - bands 3&4)
            features['speech_band_energy'] = features['band_energy_ratio'][3] + features['band_energy_ratio'][4]
            
            # High frequency energy (above 3000 Hz - bands 5+)
            features['high_freq_energy'] = sum(features['band_energy_ratio'][5:])
            
            # 4. Spectral centroid (brightness)
            features['spectral_centroid'] = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            features['normalized_centroid'] = features['spectral_centroid'] / (self.sample_rate/2)
            
            # 5. Spectral flatness (tonality)
            geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
            arithmetic_mean = np.mean(spectrum + 1e-10)
            features['spectral_flatness'] = geometric_mean / arithmetic_mean
            
            # 6. Energy envelope variance (rhythmic patterns)
            frame_size = self.sample_rate // 50  # 20 ms frames
            num_frames = len(audio_chunk) // frame_size
            energy_envelope = []
            
            for i in range(num_frames):
                frame = audio_chunk[i*frame_size:(i+1)*frame_size]
                energy = np.sum(frame**2)
                energy_envelope.append(energy)
                
            features['energy_variance'] = np.var(energy_envelope) if energy_envelope else 0
            features['num_energy_frames'] = num_frames
            
            return features
            
        except Exception as e:
            self._log_debug(f"Feature extraction error: {str(e)}")
            # Return default features on error
            return {
                'zero_crossing_rate': 0.05,
                'speech_band_energy': 0.5,
                'high_freq_energy': 0.2,
                'normalized_centroid': 0.3,
                'spectral_flatness': 0.5,
                'energy_variance': 0.1,
                'error': True
            }
    
    def _analyze_features(self, features: dict) -> Tuple[str, float, dict]:
        """
        Analyze features and determine classification
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (classification, confidence, feature_analysis)
        """
        speech_score = 0
        music_score = 0
        analysis = {}
        
        # Feature 1: Zero crossing rate
        zc_rate = features['zero_crossing_rate']
        if 0.01 < zc_rate < 0.1:
            speech_score += self.weights['zero_crossing']
            analysis['zero_crossing'] = f"Speech-like ({zc_rate:.4f})"
        elif zc_rate >= 0.1:
            music_score += self.weights['zero_crossing'] * 1.5
            analysis['zero_crossing'] = f"Music-like ({zc_rate:.4f})"
        else:
            analysis['zero_crossing'] = f"Too low ({zc_rate:.4f})"
            
        # Feature 2: Speech band energy
        speech_energy = features['speech_band_energy']
        if speech_energy > 0.4:
            speech_score += self.weights['speech_band_energy']
            analysis['speech_band'] = f"High speech energy ({speech_energy:.3f})"
        else:
            analysis['speech_band'] = f"Low speech energy ({speech_energy:.3f})"
            
        # Feature 3: High frequency energy
        high_freq = features['high_freq_energy']
        if high_freq > 0.25:
            music_score += self.weights['high_freq_energy']
            analysis['high_freq'] = f"High music energy ({high_freq:.3f})"
        else:
            analysis['high_freq'] = f"Low high-freq energy ({high_freq:.3f})"
            
        # Feature 4: Spectral centroid
        centroid = features.get('normalized_centroid', 0.3)
        if centroid > 0.2:
            music_score += self.weights['spectral_centroid']
            analysis['centroid'] = f"Bright ({centroid:.3f})"
        else:
            analysis['centroid'] = f"Dark ({centroid:.3f})"
            
        # Feature 5: Spectral flatness
        flatness = features.get('spectral_flatness', 0.5)
        if flatness < 0.1:  # More tonal (music)
            music_score += self.weights['spectral_flatness']
            analysis['flatness'] = f"Tonal/Music ({flatness:.3f})"
        elif flatness > 0.2:  # More noise-like (speech)
            speech_score += self.weights['spectral_flatness'] * 0.5
            analysis['flatness'] = f"Noisy/Speech ({flatness:.3f})"
        else:
            analysis['flatness'] = f"Neutral ({flatness:.3f})"
            
        # Feature 6: Energy variance
        energy_var = features.get('energy_variance', 0.1)
        num_frames = features.get('num_energy_frames', 0)
        if energy_var < 0.1 and num_frames > 10:
            music_score += self.weights['energy_variance']
            analysis['energy_pattern'] = f"Regular/Music ({energy_var:.3f})"
        else:
            analysis['energy_pattern'] = f"Irregular ({energy_var:.3f})"
        
        # Calculate final scores and confidence
        total_possible_speech = sum([
            self.weights['zero_crossing'],
            self.weights['speech_band_energy'],
            self.weights['spectral_flatness'] * 0.5
        ])
        
        total_possible_music = sum([
            self.weights['zero_crossing'] * 1.5,
            self.weights['high_freq_energy'],
            self.weights['spectral_centroid'],
            self.weights['spectral_flatness'],
            self.weights['energy_variance']
        ])
        
        if speech_score > music_score:
            detected_type = "speech"
            confidence = min(0.5 + (speech_score / total_possible_speech) * 0.4, 0.9)
        else:
            detected_type = "music"
            confidence = min(0.5 + (music_score / total_possible_music) * 0.4, 0.9)
            
        analysis['scores'] = {
            'speech': speech_score,
            'music': music_score,
            'speech_max': total_possible_speech,
            'music_max': total_possible_music
        }
        
        return detected_type, confidence, analysis
    
    def classify(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """
        Classify audio chunk as speech or music
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Tuple of (audio_type, confidence)
        """
        self.classification_count += 1
        
        try:
            # Convert tensor to numpy if needed
            if hasattr(audio_chunk, 'cpu'):
                audio_chunk = audio_chunk.cpu().numpy()
            elif hasattr(audio_chunk, 'numpy'):
                audio_chunk = audio_chunk.numpy()
                
            # Validate input
            if len(audio_chunk) < self.sample_rate * 0.5:
                self._log_debug(f"Audio chunk too short: {len(audio_chunk)} samples")
                return self.current_type, 0.5
                
            # Extract features
            features = self._extract_features(audio_chunk)
            
            # Handle extraction errors
            if features.get('error', False):
                self._log_debug("Using previous classification due to feature extraction error")
                return self.current_type, 0.5
            
            # Analyze features
            detected_type, confidence, analysis = self._analyze_features(features)
            
            # Debug logging
            if self.debug:
                self._log_debug(f"Classification #{self.classification_count}:")
                self._log_debug(f"  Raw detection: {detected_type} (confidence: {confidence:.3f})")
                for feature, desc in analysis.items():
                    if feature != 'scores':
                        self._log_debug(f"  {feature}: {desc}")
                scores = analysis['scores']
                self._log_debug(f"  Final scores - Speech: {scores['speech']:.1f}/{scores['speech_max']:.1f}, Music: {scores['music']:.1f}/{scores['music_max']:.1f}")
            
            # Update history
            self.history.append(detected_type)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            # Determine final classification based on history
            speech_count = self.history.count("speech")
            music_count = self.history.count("music")
            history_size = len(self.history)
            
            previous_type = self.current_type
            
            # Apply thresholds with bias toward music detection (easier to trigger)
            if speech_count >= self.speech_threshold * history_size and self.current_type != "speech":
                self.current_type = "speech"
                print(f"[CLASSIFIER] Audio type changed: {previous_type.upper()} → SPEECH (confidence: {confidence:.2f})")
                if self.debug:
                    self._log_debug(f"  History: {speech_count}/{history_size} speech classifications")
            elif music_count >= self.music_threshold * history_size and self.current_type != "music":
                self.current_type = "music"
                print(f"[CLASSIFIER] Audio type changed: {previous_type.upper()} → MUSIC (confidence: {confidence:.2f})")
                if self.debug:
                    self._log_debug(f"  History: {music_count}/{history_size} music classifications")
            
            if self.debug and self.classification_count % 10 == 0:
                self._log_debug(f"Current history: {self.history}")
                self._log_debug(f"Current type: {self.current_type}")
                
            return self.current_type, confidence
            
        except Exception as e:
            self._log_debug(f"Classification error: {str(e)}")
            return self.current_type, 0.5
    
    def set_audio_type(self, audio_type: str) -> bool:
        """
        Manually override the audio type
        
        Args:
            audio_type: "speech" or "music"
            
        Returns:
            True if successful, False if invalid type
        """
        valid_types = ["speech", "music"]
        if audio_type in valid_types:
            prev_type = self.current_type
            self.current_type = audio_type
            # Fill history with this type for persistence
            self.history = [audio_type] * self.max_history
            print(f"[CLASSIFIER] Manual override: {prev_type.upper()} → {audio_type.upper()}")
            if self.debug:
                self._log_debug(f"History reset to: {self.history}")
            return True
        else:
            self._log_debug(f"Invalid audio type: {audio_type}. Must be one of {valid_types}")
            return False
    
    def get_status(self) -> dict:
        """
        Get current classifier status
        
        Returns:
            Dictionary with current status information
        """
        return {
            'current_type': self.current_type,
            'history': self.history.copy(),
            'classification_count': self.classification_count,
            'speech_count': self.history.count("speech"),
            'music_count': self.history.count("music"),
            'weights': self.weights.copy(),
            'thresholds': {
                'speech': self.speech_threshold,
                'music': self.music_threshold,
                'min_confidence': self.min_confidence
            }
        }
    
    def reset(self):
        """Reset classifier state"""
        self.current_type = "speech"
        self.history = []
        self.classification_count = 0
        if self.debug:
            self._log_debug("Classifier state reset")

# Test/Demo functionality
def test_classifier():
    """Test the classifier with synthetic audio"""
    print("Testing Speech/Music Classifier...")
    
    # Create test classifier with debug enabled
    classifier = SpeechMusicClassifier(debug=True, max_history=3)
    
    # Generate test audio chunks
    sample_rate = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test 1: Speech-like audio (noise + some frequency content)
    print("\n" + "="*50)
    print("TEST 1: Speech-like audio")
    print("="*50)
    speech_audio = np.random.normal(0, 0.1, len(t)) + 0.3 * np.sin(2 * np.pi * 1500 * t)
    result = classifier.classify(speech_audio)
    print(f"Result: {result}")
    
    # Test 2: Music-like audio (tonal with harmonics)
    print("\n" + "="*50)
    print("TEST 2: Music-like audio")
    print("="*50)
    music_audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 
                   0.3 * np.sin(2 * np.pi * 880 * t) + 
                   0.2 * np.sin(2 * np.pi * 1320 * t))
    result = classifier.classify(music_audio)
    print(f"Result: {result}")
    
    # Test 3: Manual override
    print("\n" + "="*50)
    print("TEST 3: Manual override")
    print("="*50)
    success = classifier.set_audio_type("music")
    print(f"Manual override success: {success}")
    print(f"Status: {classifier.get_status()}")

if __name__ == "__main__":
    test_classifier()