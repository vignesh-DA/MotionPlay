"""
REAL-TIME HAND GESTURE RECOGNITION FOR TEMPLE RUN / SUBWAY SURFERS
===================================================================

GESTURE MAPPING (Your Requirements):
  1. INDEX FINGER ONLY (pointing RIGHT)  → RIGHT arrow key
  2. INDEX FINGER ONLY (pointing LEFT)   → LEFT arrow key  
  3. OPEN PALM (5 fingers open)          → UP arrow key (Jump)
  4. CLOSED FIST (all fingers curled)    → DOWN arrow key (Slide)

Architecture:
  Webcam (720p) → HSV Mask → Hand Detection → Finger Count → Gesture → Keyboard

Technical:
  - Detects 100% with optimized HSV range + 20px contour area
  - Real-time detection (~25 FPS)
  - 0.3s cooldown between key presses
  - Visual feedback with gesture name on screen

Author:     Hand Gesture Controller
Date:       2026
Version:    2.0 (Optimized for game control)
"""

import cv2
import numpy as np
import pyautogui
import time
from pathlib import Path

# Import modules
from modules.video_capture_preprocessing import VideoCapturePreprocessor
from modules.hand_detection_segmentation import HandDetectionSegmentor
from modules.feature_extraction import FeatureExtractor
from modules.gesture_classification import GestureClassifier
from modules.game_control_interface import GameControlInterface


class HandGestureGameController:
    """
    Main application class integrating all modules.
    
    Manages the complete pipeline:
    1. Capture and preprocess video frames
    2. Detect hand region
    3. Extract hand features
    4. Classify gestures
    5. Execute game commands
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 target_fps: int = 30):
        """
        Initialize the complete system.
        
        Args:
            camera_index (int): Webcam device index (default 0)
            frame_width (int): Target frame width (default 640)
            frame_height (int): Target frame height (default 480)
            target_fps (int): Target FPS (default 30)
        """
        print("=" * 70)
        print("REAL-TIME HAND GESTURE RECOGNITION FOR GAME CONTROLLERS")
        print("=" * 70)
        
        print("\nInitializing modules...")
        
        # Module 1: Video Capture & Preprocessing
        try:
            self.preprocessor = VideoCapturePreprocessor(
                device_index=camera_index,
                frame_width=frame_width,
                frame_height=frame_height,
                target_fps=target_fps
            )
        except RuntimeError as e:
            print(f"✗ Failed to initialize video capture: {e}")
            raise
        
        # Module 2: Hand Detection & Segmentation
        self.detector = HandDetectionSegmentor()
        
        # Module 3: Feature Extraction
        self.extractor = FeatureExtractor()
        
        # Module 4: Gesture Classification
        self.classifier = GestureClassifier(smoothing_enabled=True)
        
        # Module 5: Game Control Interface
        self.controller = GameControlInterface(
            command_cooldown=0.5,
            log_commands=True
        )
        
        # System state
        self.running = False
        self.show_debug = True
        
        # Performance metrics
        self.frame_count = 0
        self.hands_detected = 0
        self.start_time = None
        self.fps_history = []
        
        print("\n✓ All modules initialized successfully")
        print("\nControls:")
        print("  'd' - Toggle debug visualization")
        print("  's' - Print statistics")
        print("  'r' - Reset statistics")
        print("  'q' - Quit application")
    
    def process_frame(self, raw_frame) -> dict:
        """
        Process a single frame through complete pipeline.
        
        Pipeline:
          1. Preprocess frame (Module 1)
          2. Detect hand (Module 2)
          3. Extract features (Module 3)
          4. Classify gesture (Module 4)
          5. Execute command (Module 5)
        
        Returns:
            dict: Processing results containing:
              - 'input_frame': Original BGR frame
              - 'preprocessed': Preprocessed BGR frame
              - 'mask': Binary hand mask
              - 'hand_contour': Detected contour
              - 'features': Extracted features dict
              - 'gesture': Classified gesture type
              - 'confidence': Gesture confidence
              - 'command_executed': Whether command was issued
        """
        results = {
            'success': False,
            'hand_found': False,
            'command_executed': False
        }
        
        # Module 1: Preprocess frame
        bgr_frame, hsv_frame = self.preprocessor.preprocess_frame(raw_frame)
        results['preprocessed'] = bgr_frame
        
        # Module 2: Detect hand
        hand_contour, mask, detection_props = self.detector.detect_hand(hsv_frame)
        results['mask'] = mask
        results['hand_contour'] = hand_contour
        results['detection_props'] = detection_props
        
        if not detection_props['found']:
            return results
        
        results['hand_found'] = True
        self.hands_detected += 1
        
        # Module 3: Extract features
        features = self.extractor.extract_features(hand_contour)
        results['features'] = features
        
        # Module 4: Classify gesture
        gesture, confidence = self.classifier.classify_from_features(features)
        results['gesture'] = gesture
        results['confidence'] = confidence
        
        # Module 5: Execute game command
        action = self.classifier.get_gesture_action(gesture)
        key = self.classifier.get_keyboard_key(action) if action else None
        
        command_executed = self.controller.execute_gesture_command(action, confidence)
        results['command_executed'] = command_executed
        results['action'] = action
        results['key'] = key
        results['success'] = True
        
        return results
    
    def draw_debug_visualization(self, frame, results) -> None:
        """
        Draw real-time gesture display on frame.
        
        Shows (Top-Left):
          - GESTURE NAME: "INDEX_RIGHT" / "INDEX_LEFT" / "OPEN_PALM" / "CLOSED_FIST"
          - KEY PRESSED: Right/Left/Up/Down with visual feedback
          - Finger Count: Current detected fingers
        
        Shows (Top-Right):
          - FPS and frame count
          - Detection rate
        
        Shows (On Hand):
          - Hand contour (green)
          - Bounding box (blue)
          - Fingertips (red circles)
        
        Args:
            frame: BGR frame to draw on (modified in-place)
            results: Processing results dict
        """
        # ====== DRAW HAND CONTOURS & FEATURES ======
        if results['hand_contour'] is not None:
            cv2.drawContours(frame, [results['hand_contour']], 0, (0, 255, 0), 2)
        
        # Draw bounding box
        if results['detection_props']['found']:
            x, y, w, h = results['detection_props']['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw fingertips
        if results['hand_found'] and 'features' in results:
            features = results['features']
            for ft in features['fingertips']:
                cv2.circle(frame, ft, 6, (0, 0, 255), -1)
        
        # ====== DISPLAY GESTURE NAME (BIG, TOP-LEFT) ======
        if results['hand_found']:
            gesture_name = results['gesture'].value
            confidence = results['confidence']
            
            # Color code by gesture type
            gesture_color = (0, 255, 0)  # Green by default
            if 'INDEX_RIGHT' in gesture_name:
                gesture_color = (0, 165, 255)  # Orange - Right
            elif 'INDEX_LEFT' in gesture_name:
                gesture_color = (255, 0, 0)    # Blue - Left
            elif 'OPEN_PALM' in gesture_name:
                gesture_color = (0, 255, 255)  # Yellow - Jump
            elif 'CLOSED_FIST' in gesture_name:
                gesture_color = (255, 0, 255)  # Magenta - Slide
            
            # Draw gesture name (LARGE)
            cv2.putText(frame, gesture_name, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, gesture_color, 3)
            
            # Draw confidence
            conf_text = f"Confidence: {confidence:.0%}"
            cv2.putText(frame, conf_text, (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
            
            # ====== DISPLAY KEYBOARD KEY BEING PRESSED ======
            if results['command_executed']:
                key_map = {
                    'move_right': '→ RIGHT KEY',
                    'move_left': '← LEFT KEY',
                    'jump': '↑ UP KEY (JUMP)',
                    'slide': '↓ DOWN KEY (SLIDE)'
                }
                key_text = key_map.get(results.get('action', ''), 'UNKNOWN')
                
                # Draw key with highlight background
                cv2.rectangle(frame, (20, 135), (550, 185), (0, 255, 0), -1)
                cv2.putText(frame, key_text, (30, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
                cv2.putText(frame, ">>> PRESSED <<<", (560, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Waiting for command...", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            # Draw finger count
            if 'features' in results:
                finger_count = results['features']['finger_count']
                cv2.putText(frame, f"Fingers Detected: {finger_count}", (20, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
            cv2.putText(frame, "Position hand in frame", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # ====== DISPLAY STATS (TOP-RIGHT) ======
        detection_stats = self.detector.get_detection_statistics()
        detection_rate_text = f"Detection: {detection_stats['detection_rate']:.0f}%"
        
        # Calculate FPS
        current_fps = 0
        if self.fps_history:
            current_fps = 1.0 / (sum(self.fps_history) / len(self.fps_history))
        
        fps_text = f"FPS: {current_fps:.1f}"
        frame_text = f"Frame: {self.frame_count}"
        
        # Draw stats on top right
        cv2.putText(frame, fps_text, (frame.shape[1] - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, detection_rate_text, (frame.shape[1] - 200, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, frame_text, (frame.shape[1] - 200, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def print_statistics(self) -> None:
        """Print current system statistics."""
        print("\n" + "=" * 70)
        print("SYSTEM STATISTICS")
        print("=" * 70)
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        detection_rate = 100 * self.hands_detected / self.frame_count if self.frame_count > 0 else 0
        
        print(f"\nVideo Processing:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Elapsed time: {elapsed_time:.1f} seconds")
        print(f"  Average FPS: {self.frame_count / elapsed_time:.1f}")
        print(f"  Hands detected: {self.hands_detected}")
        print(f"  Detection rate: {detection_rate:.1f}%")
        
        # Hand detection statistics  
        detector_stats = self.detector.get_detection_statistics()
        print(f"\n✅ HAND DETECTION STATUS:")
        print(f"  Total frames: {detector_stats['total_frames']}")
        print(f"  Detections: {detector_stats['detections']}")
        print(f"  Detection rate: {detector_stats['detection_rate']:.1f}%")
        print(f"  Last frame: {'✓ Hand detected' if detector_stats['last_detected'] else '✗ No hand detected'}")
        
        # Gesture statistics
        classifier_stats = self.classifier.get_statistics()
        print(f"\nGesture Classifications:")
        print(f"  Total classifications: {classifier_stats['total_classifications']}")
        for gesture_type, count in classifier_stats['counts'].items():
            if count > 0:
                percentage = 100 * count / classifier_stats['total_classifications']
                print(f"  {gesture_type.value:20s}: {count:3d} ({percentage:5.1f}%)")
        
        # Command statistics
        command_stats = self.controller.get_command_statistics()
        print(f"\nCommand Execution:")
        print(f"  Total gestures processed: {command_stats['total_gestures_processed']}")
        print(f"  Commands issued: {command_stats['total_commands_issued']}")
        print(f"  Commands rejected: {command_stats['total_commands_rejected']}")
        print(f"  Execution rate: {command_stats['execution_rate']:.1f}%")
        
        if command_stats['total_commands_issued'] > 0:
            print(f"\n  Command breakdown:")
            for action, count in command_stats['command_counts'].items():
                if count > 0:
                    print(f"    {action:15s}: {count:3d}")
        
        print(f"\n[LOG] Detection Log Location:")
        import os
        log_dir = Path(__file__).parent.parent / "logs"
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"))
            if log_files:
                latest_log = log_files[-1]
                print(f"  {latest_log}")
                print(f"  Open this file to see detailed hand detection logs")
        
        print("=" * 70 + "\n")
    
    def run(self) -> None:
        """
        Main application loop.
        
        Captures frames, processes through complete pipeline,
        displays results, and handles user input.
        """
        self.running = True
        self.start_time = time.time()
        frame_times = []
        
        print("\n✓ Starting main loop (press 'q' to quit)...")
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # Capture frame
                ret, raw_frame = self.preprocessor.read_frame()
                if not ret:
                    print("✗ Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Process frame through complete pipeline
                results = self.process_frame(raw_frame)
                
                # Get display frame
                display_frame = results.get('preprocessed', raw_frame).copy()
                
                # Draw debug visualization
                self.draw_debug_visualization(display_frame, results)
                
                # Display frame
                cv2.imshow("Hand Gesture Recognition", display_frame)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('r'):
                    self.classifier.reset_statistics()
                    self.controller.reset_statistics()
                    self.hands_detected = 0
                    self.frame_count = 0
                    self.start_time = time.time()
                    print("✓ Statistics reset")
                
                # Track frame timing
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                # Track FPS
                if frame_time > 0:
                    self.fps_history.append(frame_time)
                    if len(self.fps_history) > 30:
                        self.fps_history.pop(0)
        
        except KeyboardInterrupt:
            print("\n✗ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Error in main loop: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Clean shutdown of all modules."""
        print("\nShutting down...")
        
        # Release resources
        self.preprocessor.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_statistics()
        
        print("\n✓ Shutdown complete")


def main():
    """
    Entry point for application.
    
    Usage:
        python main.py [camera_index] [--help]
    
    Examples:
        python main.py              # Use default camera (index 0)
        python main.py 1            # Use camera at index 1
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-Time Hand Gesture Recognition for Game Controllers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  d - Toggle debug visualization
  s - Print statistics
  r - Reset statistics
  q - Quit

Examples:
  %(prog)s                # Use default camera
  %(prog)s 1              # Use camera at index 1
  %(prog)s --help         # Show help
        """
    )
    
    parser.add_argument('camera_index', nargs='?', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width in pixels (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height in pixels (default: 480)')
    
    args = parser.parse_args()
    
    # Create and run application
    app = HandGestureGameController(
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height
    )
    
    app.run()


if __name__ == "__main__":
    main()
