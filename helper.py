#!/usr/bin/env python3
"""
Desktop Monitor Helper Utilities

Standalone helper tools for configuring and debugging the Desktop Monitor.
These utilities can be run individually and don't affect the main application.

Available tools:
1. Screen coordinate finder - helps determine monitor areas to capture
2. Audio device lister - shows available audio input devices
3. Test audio capture - verify audio is working
4. Screen capture test - verify screen capture is working

Usage:
    python helpers.py screen          # Find screen coordinates
    python helpers.py audio           # List audio devices  
    python helpers.py test-audio      # Test audio capture
    python helpers.py test-screen     # Test screen capture
    python helpers.py --help          # Show all options
"""

import argparse
import sys
import time
import numpy as np # type: ignore

def find_screen_coordinates():
    """Interactive tool to find screen coordinates for monitoring"""
    print("=" * 60)
    print("SCREEN COORDINATE FINDER")
    print("=" * 60)
    print("This tool helps you find the coordinates for specific screen areas.")
    print("You'll click on the TOP-LEFT and BOTTOM-RIGHT corners of the area you want to monitor.")
    print()
    
    try:
        import pyautogui # type: ignore
        
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        print(f"Screen resolution detected: {screen_width} x {screen_height}")
        print()
        
        # Method 1: Click to get coordinates
        print("Method 1: Click Coordinates")
        print("-" * 30)
        print("Move your mouse to the TOP-LEFT corner of the area you want to monitor...")
        input("Press Enter when ready, then you'll have 3 seconds to position your mouse...")
        
        print("Positioning... 3")
        time.sleep(1)
        print("Positioning... 2") 
        time.sleep(1)
        print("Positioning... 1")
        time.sleep(1)
        
        x1, y1 = pyautogui.position()
        print(f"Top-left coordinates captured: ({x1}, {y1})")
        print()
        
        print("Now move your mouse to the BOTTOM-RIGHT corner of the area...")
        input("Press Enter when ready, then you'll have 3 seconds to position your mouse...")
        
        print("Positioning... 3")
        time.sleep(1)
        print("Positioning... 2")
        time.sleep(1) 
        print("Positioning... 1")
        time.sleep(1)
        
        x2, y2 = pyautogui.position()
        print(f"Bottom-right coordinates captured: ({x2}, {y2})")
        print()
        
        # Calculate area
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Monitoring area coordinates:")
        print(f"  Left: {left}")
        print(f"  Top: {top}")
        print(f"  Width: {width}")
        print(f"  Height: {height}")
        print()
        print("Add this to your config.py:")
        print(f'MONITOR_AREA = {{"left": {left}, "top": {top}, "width": {width}, "height": {height}}}')
        print()
        
        # Method 2: Common presets
        print("Method 2: Common Presets")
        print("-" * 30)
        print("Here are some common monitor area configurations:")
        print()
        print("Full screen:")
        print(f'MONITOR_AREA = {{"left": 0, "top": 0, "width": {screen_width}, "height": {screen_height}}}')
        print()
        print("Center half:")
        center_left = screen_width // 4
        center_top = screen_height // 4
        center_width = screen_width // 2
        center_height = screen_height // 2
        print(f'MONITOR_AREA = {{"left": {center_left}, "top": {center_top}, "width": {center_width}, "height": {center_height}}}')
        print()
        print("Primary browser area (typical):")
        browser_left = 100
        browser_top = 100
        browser_width = screen_width - 200
        browser_height = screen_height - 200
        print(f'MONITOR_AREA = {{"left": {browser_left}, "top": {browser_top}, "width": {browser_width}, "height": {browser_height}}}')
        
    except ImportError:
        print("ERROR: pyautogui not installed.")
        print("Install it with: pip install pyautogui")
    except Exception as e:
        print(f"Error: {e}")

def list_audio_devices():
    """List all available audio input and output devices"""
    print("=" * 60)
    print("AUDIO DEVICE INFORMATION")
    print("=" * 60)
    
    try:
        import sounddevice as sd # type: ignore
        
        print("Available audio devices:")
        print()
        
        devices = sd.query_devices()
        
        # Show all devices
        for i, device in enumerate(devices):
            device_type = []
            if device['max_inputs'] > 0:
                device_type.append("INPUT")
            if device['max_outputs'] > 0:
                device_type.append("OUTPUT")
            
            type_str = "/".join(device_type) if device_type else "UNKNOWN"
            
            print(f"Device {i:2d}: {device['name']}")
            print(f"          Type: {type_str}")
            print(f"          Channels: {device['max_inputs']} in, {device['max_outputs']} out")
            print(f"          Sample Rate: {device['default_samplerate']} Hz")
            print()
        
        # Show defaults
        try:
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            print("Default devices:")
            print(f"  Input (ID {default_input}): {devices[default_input]['name']}")
            print(f"  Output (ID {default_output}): {devices[default_output]['name']}")
            print()
        except:
            print("Could not determine default devices")
            
        # Recommendations
        print("=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("For desktop audio monitoring, you typically want:")
        print("1. A 'Stereo Mix' or 'What U Hear' device to capture system audio")
        print("2. Or a virtual audio cable (VB-Audio Cable, etc.)")
        print("3. Or 'Monitor of [device]' on Linux systems")
        print()
        print("If you don't see system audio capture options:")
        print("- Windows: Enable 'Stereo Mix' in Sound settings")
        print("- macOS: Use tools like BlackHole or SoundFlower") 
        print("- Linux: Use PulseAudio monitor devices")
        
    except ImportError:
        print("ERROR: sounddevice not installed.")
        print("Install it with: pip install sounddevice")
    except Exception as e:
        print(f"Error: {e}")

def test_audio_capture():
    """Test audio capture to verify it's working"""
    print("=" * 60)
    print("AUDIO CAPTURE TEST")
    print("=" * 60)
    
    try:
        import sounddevice as sd # type: ignore
        import numpy as np # type: ignore
        
        # Get device info
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        print(f"Testing audio capture from device {default_input}:")
        print(f"  {devices[default_input]['name']}")
        print()
        print("Starting 10-second audio capture test...")
        print("Make some noise to test (play music, talk, etc.)")
        print()
        
        # Record for 10 seconds
        sample_rate = 44100
        duration = 10
        
        print("Recording... (speak or play audio now)")
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()
        
        # Analyze the captured audio
        audio_data = audio_data.flatten()
        
        # Calculate statistics
        rms_level = np.sqrt(np.mean(audio_data**2))
        max_level = np.max(np.abs(audio_data))
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)) != 0)
        
        print("Capture complete!")
        print()
        print("Audio Analysis:")
        print(f"  RMS Level: {rms_level:.6f}")
        print(f"  Peak Level: {max_level:.6f}")
        print(f"  Zero Crossings: {zero_crossings}")
        print()
        
        # Interpretation
        if rms_level < 0.0001:
            print("⚠️  WARNING: Very low audio level detected")
            print("   - Check if the correct input device is selected")
            print("   - Verify audio is actually playing")
            print("   - Check system volume and input levels")
        elif rms_level < 0.001:
            print("⚠️  CAUTION: Low audio level")
            print("   - Audio detected but quite quiet")
            print("   - May need to increase input gain")
        else:
            print("✅ GOOD: Audio levels detected successfully")
            print("   - Desktop Monitor should work with this audio")
            
        print()
        print("Recommended config.py settings:")
        if rms_level > 0.01:
            print("# High audio levels - use stricter noise gate")
            print("NOISE_THRESHOLD = 0.001")
        elif rms_level > 0.001:
            print("# Normal audio levels")  
            print("NOISE_THRESHOLD = 0.0005")
        else:
            print("# Low audio levels - use sensitive noise gate")
            print("NOISE_THRESHOLD = 0.0003")
            
    except ImportError:
        print("ERROR: sounddevice not installed.")
        print("Install it with: pip install sounddevice")
    except Exception as e:
        print(f"Error during audio test: {e}")

def test_screen_capture():
    """Test screen capture to verify it's working"""
    print("=" * 60)
    print("SCREEN CAPTURE TEST")
    print("=" * 60)
    
    try:
        from mss import mss # type: ignore
        import numpy as np # type: ignore
        from PIL import Image # type: ignore
        import os
        
        print("Testing screen capture...")
        
        with mss() as sct:
            # Get monitor info
            monitors = sct.monitors
            print(f"Detected {len(monitors)-1} monitor(s):")
            
            for i, monitor in enumerate(monitors[1:], 1):  # Skip 'All in One' monitor
                print(f"  Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
            
            print()
            
            # Test capture from primary monitor
            primary_monitor = monitors[1]  # First real monitor
            print(f"Testing capture from primary monitor...")
            print(f"Area: {primary_monitor}")
            
            # Capture screenshot
            screenshot = sct.grab(primary_monitor)
            
            # Convert to PIL Image
            img = Image.frombytes('RGB', 
                                (screenshot.width, screenshot.height), 
                                screenshot.rgb)
            
            # Basic analysis
            img_array = np.array(img)
            
            # Calculate statistics
            mean_brightness = np.mean(img_array)
            brightness_std = np.std(img_array)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            
            print("✅ Screen capture successful!")
            print()
            print("Capture Analysis:")
            print(f"  Resolution: {screenshot.width} x {screenshot.height}")
            print(f"  Mean brightness: {mean_brightness:.1f}")
            print(f"  Brightness variation: {brightness_std:.1f}")
            print(f"  Unique colors: {unique_colors}")
            
            # Save test image
            test_filename = "screen_capture_test.png"
            img.save(test_filename)
            print(f"  Test image saved as: {test_filename}")
            print()
            
            # Recommendations
            if brightness_std < 10:
                print("⚠️  WARNING: Very low variation in screen content")
                print("   - Make sure screen isn't blank or showing static content")
                print("   - Try opening a web browser or application")
            elif unique_colors < 100:
                print("⚠️  CAUTION: Limited color variation detected")
                print("   - Screen may be showing simple content")
            else:
                print("✅ GOOD: Rich screen content detected")
                print("   - Desktop Monitor should work well")
            
            print()
            print("Recommended config.py settings:")
            print(f'MONITOR_AREA = {{"left": {primary_monitor["left"]}, "top": {primary_monitor["top"]}, "width": {primary_monitor["width"]}, "height": {primary_monitor["height"]}}}')
            
            # Test change detection
            print()
            print("Testing change detection... (move mouse or switch windows)")
            input("Press Enter to capture second screenshot for comparison...")
            
            screenshot2 = sct.grab(primary_monitor)
            img2 = Image.frombytes('RGB', 
                                 (screenshot2.width, screenshot2.height), 
                                 screenshot2.rgb)
            
            # Calculate difference
            diff = np.abs(np.array(img2).astype(int) - np.array(img).astype(int))
            change_percent = np.count_nonzero(diff) / diff.size
            
            print(f"Change detected: {change_percent:.4f} ({change_percent*100:.2f}%)")
            
            if change_percent > 0.1:
                print("✅ GOOD: Significant change detected")
            elif change_percent > 0.01:
                print("✅ OK: Moderate change detected")
            else:
                print("⚠️  Low change detected - try making more obvious changes")
                
    except ImportError as e:
        missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
        print(f"ERROR: {missing_module} not installed.")
        if missing_module == "mss":
            print("Install it with: pip install mss")
        elif missing_module == "PIL":
            print("Install it with: pip install Pillow")
    except Exception as e:
        print(f"Error during screen test: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Desktop Monitor Helper Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  screen          Find screen coordinates for monitoring
  audio           List available audio devices
  test-audio      Test audio capture functionality  
  test-screen     Test screen capture functionality

Examples:
  python helpers.py screen
  python helpers.py audio
  python helpers.py test-audio
  python helpers.py test-screen
        """
    )
    
    parser.add_argument('command', 
                       choices=['screen', 'audio', 'test-audio', 'test-screen'],
                       help='Helper command to run')
    
    args = parser.parse_args()
    
    if args.command == 'screen':
        find_screen_coordinates()
    elif args.command == 'audio':
        list_audio_devices()
    elif args.command == 'test-audio':
        test_audio_capture()
    elif args.command == 'test-screen':
        test_screen_capture()

if __name__ == "__main__":
    main()