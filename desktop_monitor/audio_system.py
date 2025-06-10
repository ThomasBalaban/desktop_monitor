"""
Desktop Audio Monitoring System

Captures and transcribes desktop audio using Whisper
"""

import os
import time
import threading
import numpy as np # type: ignore
import sounddevice as sd # type: ignore
import soundfile as sf # type: ignore
from queue import Queue # type: ignore
from threading import Thread, Event # type: ignore
from scipy import signal # type: ignore

# Import the standalone classifier
from .speech_music_classifier import SpeechMusicClassifier

class AudioProcessor:
    """Processes audio chunks for transcription"""
    
    def __init__(self, transcriber):
        self.transcriber = transcriber
        
    def process_chunk(self, chunk):
        """Process audio chunk and transcribe"""
        try:
            if self.transcriber.stop_event.is_set():
                self.transcriber.active_threads -= 1
                return
                
            # Pre-process audio
            if len(chunk) < self.transcriber.config.FS * 0.5:
                self.transcriber.active_threads -= 1
                return
                
            # Apply noise gate
            amplitude = np.abs(chunk).mean()
            if amplitude < 0.005:
                self.transcriber.active_threads -= 1
                return
                
            # Apply compression and normalization
            threshold = 0.02
            ratio = 0.5
            compressed = np.zeros_like(chunk)
            for i in range(len(chunk)):
                if abs(chunk[i]) > threshold:
                    compressed[i] = chunk[i]
                else:
                    compressed[i] = chunk[i] * ratio
            
            max_val = np.max(np.abs(compressed))
            if max_val < 1e-10:
                self.transcriber.active_threads -= 1
                return
                
            chunk = compressed / max_val
            
            # Ensure even length
            if len(chunk) % 2 != 0:
                chunk = np.pad(chunk, (0, 1), 'constant')
            
            # Save audio file if keeping files
            filename = None
            if self.transcriber.config.KEEP_AUDIO_FILES:
                filename = self.save_audio(chunk)
            
            # Classify audio type
            if self.transcriber.config.AUTO_DETECT_AUDIO_TYPE:
                audio_type, confidence = self.transcriber.classifier.classify(chunk)
            else:
                audio_type = self.transcriber.classifier.current_type
                confidence = 0.8
                
            # Skip if confidence too low
            if confidence < 0.4:
                self.transcriber.active_threads -= 1
                if filename and os.path.exists(filename):
                    os.remove(filename)
                return
                
            # Set transcription parameters based on audio type
            if audio_type == "speech":
                params = {
                    "fp16": (self.transcriber.config.DEVICE == "cuda"),
                    "beam_size": 1,
                    "temperature": 0.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_previous_text": False
                }
            else:  # music
                params = {
                    "fp16": (self.transcriber.config.DEVICE == "cuda"),
                    "beam_size": 1,
                    "temperature": 0.3,
                    "no_speech_threshold": 0.3,
                    "condition_on_previous_text": False
                }
            
            # Transcribe
            try:
                result = self._transcribe_audio(chunk, params)
                text = result.get("text", "").strip()
                
                # Remove repetitive patterns
                import re
                text = re.sub(r'(\w)(\s*-\s*\1){3,}', r'\1...', text)
                
                min_length = 2 if audio_type == "speech" else 4
                
                if text and len(text) >= min_length:
                    self.transcriber.result_queue.put((text, filename, audio_type, confidence))
                else:
                    if filename and os.path.exists(filename):
                        os.remove(filename)
                    
            except Exception as e:
                print(f"[AUDIO] Transcription error: {str(e)}")
                if filename and os.path.exists(filename):
                    os.remove(filename)
                
        except Exception as e:
            print(f"[AUDIO] Processing error: {str(e)}")
        finally:
            self.transcriber.active_threads -= 1
            
    def _transcribe_audio(self, chunk, params):
        """Handle transcription with error handling"""
        whisper_input = chunk.astype(np.float32)
        
        if whisper_input.size == 0:
            raise ValueError("Empty audio chunk")
            
        if np.isnan(whisper_input).any() or np.isinf(whisper_input).any():
            raise ValueError("Audio contains NaN or Inf values")
        
        # Apply low-pass filter
        try:
            nyquist = self.transcriber.config.FS / 2.0
            cutoff = min(8000 / nyquist, 0.99)
            b, a = signal.butter(5, cutoff, 'low')
            whisper_input = signal.filtfilt(b, a, whisper_input)
            whisper_input = whisper_input.astype(np.float32)
        except Exception as e:
            print(f"[AUDIO] Filter error: {str(e)}, skipping filtering")
            
        # Ensure sample rate is 16kHz for Whisper
        if self.transcriber.config.FS != 16000:
            try:
                import librosa # type: ignore
                whisper_input = librosa.resample(whisper_input, orig_sr=self.transcriber.config.FS, target_sr=16000)
                whisper_input = whisper_input.astype(np.float32)
            except ImportError:
                number_of_samples = round(len(whisper_input) * 16000 / self.transcriber.config.FS)
                whisper_input = signal.resample(whisper_input, number_of_samples)
                whisper_input = whisper_input.astype(np.float32)
        
        if whisper_input.size == 0:
            raise ValueError("Resampled audio is empty")
        
        whisper_input = np.ascontiguousarray(whisper_input, dtype=np.float32)
        
        # Check content energy
        content_energy = np.sqrt(np.mean(whisper_input**2))
        if content_energy < 0.01:
            raise ValueError("Processed audio too quiet")
        
        # Convert to tensor for transcription
        import torch # type: ignore
        if not torch.is_tensor(whisper_input):
            whisper_input = torch.tensor(whisper_input, dtype=torch.float32)
        
        try:
            result = self.transcriber.model.transcribe(whisper_input, **params)
            return result
        except RuntimeError as e:
            print(f"[AUDIO] Transcription runtime error: {str(e)}")
            # Try with simpler parameters
            basic_params = {
                "fp16": False,
                "beam_size": 1,
                "temperature": 0.0,
                "condition_on_previous_text": False
            }
            result = self.transcriber.model.transcribe(whisper_input, **basic_params)
            return result
            
    def save_audio(self, chunk):
        """Save audio chunk to file"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.transcriber.config.SAVE_DIR, f"desktop_{timestamp}.wav")
        sf.write(filename, chunk, self.transcriber.config.FS, subtype='PCM_16')
        return filename

class DesktopAudioMonitor:
    """Main desktop audio monitoring class"""
    
    def __init__(self, config):
        self.config = config
        self.result_queue = Queue()
        self.stop_event = Event()
        self.active_threads = 0
        self.processing_lock = Event()
        self.processing_lock.set()
        self.last_processed = time.time()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_noise_log = 0
        
        # Initialize Whisper model
        print(f"[AUDIO] Loading Whisper model: {config.MODEL_SIZE} on {config.DEVICE}")
        try:            
            import whisper # type: ignore
            self.model = whisper.load_model(config.MODEL_SIZE)
            
            if config.DEVICE == "cuda" and hasattr(self.model, "to"):
                self.model = self.model.to(config.DEVICE)
                self.model = self.model.half()
                
        except Exception as e:
            print(f"[AUDIO] Error loading model: {e}")
            raise
            
        # Initialize classifier and processor
        self.classifier = SpeechMusicClassifier(
            debug=config.DEBUG_CLASSIFIER if hasattr(config, 'DEBUG_CLASSIFIER') else False,
            sample_rate=config.FS
        )
        self.processor = AudioProcessor(self)
        
    def audio_callback(self, indata, frames, timestamp, status):
        """Process incoming audio data"""
        try:
            if not self.processing_lock.is_set() and self.active_threads < self.config.MAX_THREADS * 0.5:
                self.processing_lock.set()
                    
            if status:
                print(f"[AUDIO] Audio status: {status}")
            
            if self.stop_event.is_set():
                return
                
            new_audio = np.squeeze(indata).astype(np.float32)
            
            # Noise gate
            rms_amplitude = np.sqrt(np.mean(new_audio**2))
            if rms_amplitude < 0.0003:
                if time.time() - self.last_noise_log > 5:
                    print(f"[AUDIO] Input too quiet: {rms_amplitude:.6f} RMS, skipping")
                    self.last_noise_log = time.time()
                return
                
            # Handle invalid data
            if np.isnan(new_audio).any() or np.isinf(new_audio).any():
                print("[AUDIO] Warning: Input contains NaN or Inf values, replacing with zeros")
                new_audio = np.nan_to_num(new_audio, nan=0.0, posinf=0.0, neginf=0.0)
                
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
            
            # Limit buffer size
            max_buffer_size = self.config.FS * 30
            if len(self.audio_buffer) > max_buffer_size:
                self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                print("[AUDIO] Warning: Audio buffer too large, trimming")
            
            # Process when buffer reaches target duration
            if (self.processing_lock.is_set() and 
                len(self.audio_buffer) >= self.config.FS * self.config.CHUNK_DURATION and
                self.active_threads < self.config.MAX_THREADS):
                    
                buffer_energy = np.abs(self.audio_buffer[:int(self.config.FS*self.config.CHUNK_DURATION)]).mean()
                if buffer_energy < 0.0005:
                    self.audio_buffer = self.audio_buffer[int(self.config.FS*(self.config.CHUNK_DURATION-self.config.OVERLAP)):]
                    return
                    
                chunk = self.audio_buffer[:int(self.config.FS*self.config.CHUNK_DURATION)].copy()
                self.audio_buffer = self.audio_buffer[int(self.config.FS*(self.config.CHUNK_DURATION-self.config.OVERLAP)):]
                    
                self.active_threads += 1
                Thread(target=self.processor.process_chunk, args=(chunk,)).start()
                self.last_processed = time.time()
                
                if self.active_threads >= self.config.MAX_THREADS * 0.8:
                    self.processing_lock.clear()
                    print(f"[AUDIO] Pausing processing - too many active threads: {self.active_threads}")
                    
        except Exception as e:
            print(f"[AUDIO] Audio callback error: {e}")
            self.audio_buffer = np.array([], dtype=np.float32)
            
    def output_worker(self):
        """Process and display transcription results"""
        while not self.stop_event.is_set():
            try:
                if not self.result_queue.empty():
                    text, filename, audio_type, confidence = self.result_queue.get()
                    latency = time.time() - self.last_processed
                    timestamp = time.strftime("%H:%M:%S")
                    
                    if text:
                        print(f"[{timestamp} | {latency:.1f}s] [DESKTOP {audio_type.upper()} {confidence:.2f}] {text}")
                    
                    # Clean up file
                    if filename and os.path.exists(filename) and not self.config.KEEP_AUDIO_FILES:
                        try:
                            os.remove(filename)
                        except Exception as e:
                            print(f"[AUDIO] Error removing file: {str(e)}")
                    
                    self.result_queue.task_done()
                time.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO] Output worker error: {str(e)}")
        
    def start(self):
        """Start desktop audio monitoring"""
        print(f"[AUDIO] Model: {self.config.MODEL_SIZE.upper()} | Device: {self.config.DEVICE.upper()}")
        print(f"[AUDIO] Chunk: {self.config.CHUNK_DURATION}s with {self.config.OVERLAP}s overlap")

        # Start output worker thread
        output_thread = Thread(target=self.output_worker, daemon=True)
        output_thread.start()
        
        try:
            # Start audio input stream
            with sd.InputStream(
                samplerate=self.config.FS,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.config.FS//10
            ):
                print("[AUDIO] Listening to desktop audio...")
                
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"[AUDIO] Error in audio stream: {e}")
        finally:
            self.stop_event.set()
            
    def stop(self):
        """Stop desktop audio monitoring"""
        print("[AUDIO] Stopping desktop audio monitoring...")
        self.stop_event.set()
        
        # Clean up audio files if not keeping them
        if not self.config.KEEP_AUDIO_FILES:
            try:
                for filename in os.listdir(self.config.SAVE_DIR):
                    if filename.startswith("desktop_"):
                        filepath = os.path.join(self.config.SAVE_DIR, filename)
                        os.remove(filepath)
                print("[AUDIO] Cleaned up audio files")
            except Exception as e:
                print(f"[AUDIO] Error cleaning up files: {e}")