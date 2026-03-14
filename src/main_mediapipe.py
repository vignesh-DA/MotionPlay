"""
REAL-TIME HAND GESTURE RECOGNITION - MEDIAPIPE VERSION
========================================================

Uses MediaPipe Hands (solutions API) for 99%+ accurate hand detection + 21 keypoints.
Classifies 5 gestures for Temple Run / Subway Surfers game control.

GESTURE MAPPING (v3.0 - 5 Gesture System):
  1. OPEN_PALM        → NO ACTION (neutral - straight path)
  2. THUMBS_UP        → UP arrow key (jump)
  3. CLOSED_FIST      → DOWN arrow key (slide/duck)
  4. Point LEFT       → LEFT arrow key (move left)
  5. Point RIGHT      → RIGHT arrow key (move right)

HOW TO USE:
  Straight: Open your hand (all fingers spread) - no key pressed
  Jump:     Thumbs up (only thumb raised)
  Slide:    Close your fist (all fingers curled)
  Move L:   Point your index finger to the LEFT
  Move R:   Point your index finger to the RIGHT

KEY ADVANTAGES:
  ✓ 5 distinct gestures for full game control
  ✓ OPEN_PALM = neutral (no accidental key presses on straight paths)
  ✓ Index finger direction for left/right
  ✓ Works in ANY lighting condition
  ✓ Real-time 30+ FPS on CPU

Author:     Hand Gesture Controller v3.0
Date:       2026
Framework:  MediaPipe (solutions.hands API)
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from enum import Enum
from collections import deque
import os
import logging
from datetime import datetime

# Try new Tasks API first, fall back to solutions API
try:
    from mediapipe.tasks import vision
    USE_TASKS_API = True
except ImportError:
    # Use old solutions API
    USE_TASKS_API = False
    print("Using MediaPipe solutions.hands API")

# Setup logging with UTF-8 encoding for Windows compatibility
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f"{log_dir}/gesture_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create handlers with explicit UTF-8 encoding
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
console_handler = logging.StreamHandler()
console_handler.setStream(__import__('sys').stdout)

# Set encoding for console handler (Windows compatibility)
if hasattr(console_handler, 'setEncoding'):
    console_handler.setEncoding('utf-8')

# Create formatter
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class GestureType(Enum):
    """Gesture classification enum."""
    OPEN_PALM = "OPEN_PALM"       # Neutral - no action
    THUMBS_UP = "THUMBS_UP"       # Jump (UP key)
    CLOSED_FIST = "CLOSED_FIST"   # Slide (DOWN key)
    INDEX_RIGHT = "INDEX_RIGHT"   # Move right (RIGHT key)
    INDEX_LEFT = "INDEX_LEFT"     # Move left (LEFT key)
    UNDEFINED = "UNDEFINED"


class HandGestureRecognizer:
    """
    MediaPipe-based hand gesture recognizer for game control.
    
    Uses 21 hand landmarks to classify gestures in real-time.
    Compatible with all MediaPipe versions (uses solutions.hands API).
    """
    
    def _ensure_model(self):
        """Download hand_landmarker.task model if not present (Tasks API only)."""
        if not USE_TASKS_API:
            return  # No model needed for solutions API
        
        model_file = 'hand_landmarker.task'
        
        if os.path.exists(model_file):
            print(f"✓ Model found: {model_file}")
            return
        
        print(f"Downloading model: {model_file}...")
        import urllib.request
        
        model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        
        try:
            urllib.request.urlretrieve(model_url, model_file)
            print(f"✓ Model downloaded: {model_file}")
        except Exception as e:
            print(f"ERROR: Failed to download model: {e}")
            raise
    
    def __init__(self):
        """Initialize MediaPipe Hand Detector (uses solutions API for compatibility)."""
        logger.info("="*70)
        logger.info("HAND GESTURE RECOGNITION - INITIALIZATION")
        logger.info("="*70)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hand detector (lowered thresholds for game play robustness)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        logger.info("✓ MediaPipe Hand Detector initialized")
        logger.info(f"  - Detection confidence: 0.5 (lowered for gameplay)")
        logger.info(f"  - Tracking confidence: 0.3 (lowered for gameplay)")
        logger.info(f"  - Max hands: 1")
        
        # Gesture smoothing (temporal filtering)
        self.gesture_history = deque(maxlen=5)
        self.last_gesture = GestureType.UNDEFINED
        self.last_command_time = 0
        self.command_cooldown = 0.15  # seconds (faster response for gameplay)
        
        logger.info(f"  - Command cooldown: {self.command_cooldown}s (reduced for gameplay)")
        logger.info(f"  - Gesture smoothing: Instant (1-frame) for faster response")
        
        # Statistics
        self.frame_count = 0
        self.hands_detected = 0
        self.gesture_counts = {g.value: 0 for g in GestureType}
        self.command_counts = {
            'move_right': 0,
            'move_left': 0,
            'jump': 0,
            'slide': 0
        }
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("✓ Video capture initialized")
        logger.info(f"  - Resolution: 1280x720")
        logger.info(f"  - Target FPS: 30")
        logger.info(f"  - Log file: {log_filename}")
        logger.info("="*70)
        logger.info("")
    
    def detect_hand(self, frame):
        """
        Detect hand landmarks in frame using MediaPipe Hands.
        
        Returns:
            tuple: (landmarks_pos, handedness) or (None, None) if no hand found
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None, None
        
        # Extract landmarks from first (only) detected hand
        landmarks = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        
        landmarks_pos = np.zeros((21, 2), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark):
            landmarks_pos[i] = [lm.x * w, lm.y * h]
        
        handedness = results.multi_handedness[0].classification[0].label
        
        return landmarks_pos, handedness
    
    def count_fingers(self, landmarks_pos):
        """
        Count raised fingers using landmark positions.
        
        MediaPipe Hand Landmarks:
        - Thumb: 0=wrist, 1=CMC, 2=MCP, 3=IP, 4=tip
        - Index: 5=MCP, 6=PIP, 7=DIP, 8=tip
        - Middle: 9=MCP, 10=PIP, 11=DIP, 12=tip
        - Ring: 13=MCP, 14=PIP, 15=DIP, 16=tip
        - Pinky: 17=MCP, 18=PIP, 19=DIP, 20=tip
        
        Finger is "raised" if DIP (knuckle) is HIGHER than MCP (lower Y value).
        Uses comparison: DIP.y < MCP.y means finger is extended (raised)
        
        Returns:
            tuple: (raised_count, detailed_measurements) 
                   detailed_measurements = list of dicts with finger measurements
        """
        # For each finger, compare DIP position to MPC position
        # If DIP is higher (smaller Y), finger is raised
        # If DIP is lower (larger Y), finger is closed
        
        fingers = [
            (4, 3, "Thumb"),      # Thumb: tip=4, IP=3
            (8, 7, "Index"),      # Index: tip=8, DIP=7  
            (12, 11, "Middle"),   # Middle: tip=12, DIP=11
            (16, 15, "Ring"),     # Ring: tip=16, DIP=15
            (20, 19, "Pinky"),    # Pinky: tip=20, DIP=19
        ]
        
        raised_count = 0
        measurements = []
        margin = 3  # pixels threshold
        
        for tip_idx, dip_idx, finger_name in fingers:
            tip_y = landmarks_pos[tip_idx][1]
            dip_y = landmarks_pos[dip_idx][1]
            distance = dip_y - tip_y  # positive if DIP is below tip
            is_raised = tip_y < dip_y - margin
            
            if is_raised:
                raised_count += 1
            
            measurements.append({
                'finger': finger_name,
                'tip_y': int(tip_y),
                'dip_y': int(dip_y),
                'distance': int(distance),
                'margin': margin,
                'is_raised': is_raised
            })
        
        return raised_count, measurements
    
    def classify_gesture(self, landmarks_pos, frame_shape):
        """
        Classify hand gesture from landmarks.
        
        GESTURE LOGIC (v3.0 - 5 Gesture System):
          - ≥3 non-thumb fingers raised → OPEN_PALM (neutral, no action)
          - 0 non-thumb fingers + thumb raised → THUMBS_UP (jump/up)
          - 0 non-thumb fingers + thumb closed → CLOSED_FIST (slide/down)
          - 1-2 non-thumb fingers → check index finger direction (left/right)
          
        Separates thumb from other fingers for reliable detection.
        
        Args:
            landmarks_pos: (21, 2) array of landmark positions
            frame_shape: (height, width) of frame
            
        Returns:
            tuple: (GestureType, finger_count, measurements)
        """
        h, w = frame_shape[:2]
        finger_count, measurements = self.count_fingers(landmarks_pos)
        
        # Separate thumb from non-thumb fingers
        thumb_raised = measurements[0]['is_raised']  # Thumb is first in measurements
        non_thumb_raised = sum(1 for m in measurements[1:] if m['is_raised'])
        
        # OPEN_PALM: 3+ non-thumb fingers raised → NEUTRAL (no action)
        if non_thumb_raised >= 3:
            return GestureType.OPEN_PALM, finger_count, measurements
        
        # No non-thumb fingers raised → check thumb
        if non_thumb_raised == 0:
            if thumb_raised:
                # THUMBS_UP: Only thumb is raised → JUMP (UP key)
                return GestureType.THUMBS_UP, finger_count, measurements
            else:
                # CLOSED_FIST: Everything closed → SLIDE (DOWN key)
                return GestureType.CLOSED_FIST, finger_count, measurements
        
        # 1-2 non-thumb fingers raised → LEFT/RIGHT by index finger direction
        index_tip_x = landmarks_pos[8][0]   # Index fingertip
        index_mcp_x = landmarks_pos[5][0]   # Index knuckle (MCP)
        
        if index_tip_x < index_mcp_x:
            return GestureType.INDEX_LEFT, finger_count, measurements
        else:
            return GestureType.INDEX_RIGHT, finger_count, measurements
    
    def smooth_gesture(self, current_gesture):
        """
        Gesture smoothing for gameplay (instant response with stability).
        
        For gameplay: Instantly confirms valid gestures (0 frame latency)
        For stability: Holds last valid gesture when hand is lost (UNDEFINED)
        
        Args:
            current_gesture: GestureType from current frame
            
        Returns:
            GestureType: Smooth gesture (instant response, prevents jitter)
        """
        self.gesture_history.append(current_gesture)
        
        # Instant confirmation for valid gestures (0 frame latency for gameplay)
        if current_gesture != GestureType.UNDEFINED:
            return current_gesture
        
        # Hand lost: hold previous gesture briefly to prevent jitter
        return self.last_gesture
    
    def execute_command(self, gesture):
        """
        Execute keyboard command for gesture.
        
        Args:
            gesture: GestureType to execute
            
        Returns:
            bool: True if command executed, False if on cooldown
        """
        if gesture == GestureType.UNDEFINED:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return False
        
        # OPEN_PALM = neutral, no key press
        if gesture == GestureType.OPEN_PALM:
            return False
        
        # Map gesture to keyboard command
        gesture_to_key = {
            GestureType.THUMBS_UP: 'up',
            GestureType.CLOSED_FIST: 'down',
            GestureType.INDEX_RIGHT: 'right',
            GestureType.INDEX_LEFT: 'left'
        }
        
        if gesture in gesture_to_key:
            key = gesture_to_key[gesture]
            
            try:
                pyautogui.press(key)
                self.last_command_time = current_time
                
                # Track command
                action_map = {
                    'right': 'move_right',
                    'left': 'move_left',
                    'up': 'jump',
                    'down': 'slide'
                }
                if key in action_map:
                    action = action_map[key]
                    self.command_counts[action] += 1
                    logger.info(f"Frame {self.frame_count}: [COMMAND] {gesture.value} → {key.upper()} key ({action})")
                
                return True
            except Exception as e:
                print(f"Error executing command: {e}")
                return False
        
        return False
    
    def draw_landmarks(self, frame, landmarks_pos):
        """Draw hand landmarks and skeleton on frame."""
        # Draw skeleton connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        for start, end in connections:
            x1, y1 = landmarks_pos[start].astype(int)
            x2, y2 = landmarks_pos[end].astype(int)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw finger tips (landmarks 4, 8, 12, 16, 20)
        fingertip_indices = [4, 8, 12, 16, 20]
        for idx in fingertip_indices:
            x, y = landmarks_pos[idx].astype(int)
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
        
        # Draw wrist and other landmarks
        for i, (x, y) in enumerate(landmarks_pos):
            if i == 0:  # Wrist
                cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 0), -1)
            else:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 0), -1)
    
    def draw_gesture_display(self, frame, gesture, confidence=1.0):
        """
        Draw gesture name and keyboard command on frame.
        
        Args:
            frame: Frame to draw on
            gesture: GestureType to display
            confidence: Confidence level (0-1)
        """
        h, w = frame.shape[:2]
        
        # Gesture name with color coding
        gesture_color_map = {
            GestureType.OPEN_PALM: (0, 255, 0),        # Green - Neutral (no action)
            GestureType.THUMBS_UP: (0, 255, 255),      # Yellow - Jump
            GestureType.CLOSED_FIST: (255, 0, 255),    # Magenta - Slide
            GestureType.INDEX_RIGHT: (0, 165, 255),    # Orange - Right
            GestureType.INDEX_LEFT: (255, 0, 0),       # Blue - Left
            GestureType.UNDEFINED: (128, 128, 128)     # Gray
        }
        
        color = gesture_color_map.get(gesture, (128, 128, 128))
        
        # Draw gesture name (BIG)
        gesture_text = gesture.value
        cv2.putText(frame, gesture_text, (30, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
        
        # Draw confidence
        conf_text = f"Conf: {confidence:.0%}"
        cv2.putText(frame, conf_text, (30, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw keyboard command
        gesture_to_display = {
            GestureType.OPEN_PALM: '— NEUTRAL (no key)',
            GestureType.THUMBS_UP: '↑ UP KEY (JUMP)',
            GestureType.CLOSED_FIST: '↓ DOWN KEY (SLIDE)',
            GestureType.INDEX_RIGHT: '→ RIGHT KEY',
            GestureType.INDEX_LEFT: '← LEFT KEY'
        }
        
        if gesture in gesture_to_display:
            cmd_text = gesture_to_display[gesture]
            
            # Draw command with background
            cv2.rectangle(frame, (30, 160), (450, 210), color, -1)
            cv2.putText(frame, cmd_text, (50, 195),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Draw stats (top right)
        fps_text = f"Frame: {self.frame_count}"
        detection_rate = 100 * self.hands_detected / max(1, self.frame_count)
        detection_text = f"Detection: {detection_rate:.0f}%"
        
        cv2.putText(frame, fps_text, (w - 300, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, detection_text, (w - 300, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw finger count
        if hasattr(self, 'last_landmarks') and self.last_landmarks is not None:
            finger_count, _ = self.count_fingers(self.last_landmarks)
        else:
            finger_count = 0
        finger_text = f"Fingers: {finger_count}"
        cv2.putText(frame, finger_text, (30, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def run(self):
        """Main processing loop."""
        logger.info("")
        logger.info("="*70)
        logger.info("MEDIAPIPE HAND GESTURE RECOGNITION - GAME CONTROLLER")
        logger.info("="*70)
        logger.info("")
        logger.info("GESTURE MAPPING (v3.0 - 5 Gesture System):")
        logger.info("  ✓ OPEN_PALM     → NO ACTION (neutral/straight path)")
        logger.info("  ✓ THUMBS_UP     → UP arrow key (Jump)")
        logger.info("  ✓ CLOSED_FIST   → DOWN arrow key (Slide)")
        logger.info("  ✓ Point LEFT    → LEFT arrow key (Move left)")
        logger.info("  ✓ Point RIGHT   → RIGHT arrow key (Move right)")
        logger.info("")
        logger.info("CONTROLS (during execution):")
        logger.info("  'q' - Quit application")
        logger.info("  's' - Show statistics")
        logger.info("")
        logger.info("Starting real-time hand detection...")
        logger.info("="*70)
        logger.info("")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Flip frame horizontally (mirror)
                frame = cv2.flip(frame, 1)
                
                # Detect hand
                landmarks, handedness = self.detect_hand(frame)
                
                if landmarks is not None:
                    self.hands_detected += 1
                    h, w = frame.shape[:2]
                    
                    # landmarks is already (21, 2) array from detect_hand
                    landmarks_pos = landmarks
                    self.last_landmarks = landmarks_pos
                    
                    # Classify gesture (now returns gesture, finger_count, measurements)
                    current_gesture, finger_count, measurements = self.classify_gesture(landmarks_pos, frame.shape)
                    
                    # Smooth gesture
                    smoothed_gesture = self.smooth_gesture(current_gesture)
                    self.last_gesture = smoothed_gesture
                    
                    # Track gesture counts
                    self.gesture_counts[smoothed_gesture.value] += 1
                    
                    # Log every 30 frames with detailed measurements
                    if self.frame_count % 30 == 0:
                        # Format detailed measurements for logging
                        measurements_str = " | ".join([
                            f"{m['finger']}: tip={m['tip_y']} dip={m['dip_y']} dist={m['distance']} {'✓RAISED' if m['is_raised'] else '✗closed'}"
                            for m in measurements
                        ])
                        logger.info(f"Frame {self.frame_count}: [OK] DETECTED - Gesture: {smoothed_gesture.value} (fingers: {finger_count})")
                        logger.info(f"  └─ Y-Coordinate Distances (margin={measurements[0]['margin']}px): {measurements_str}")
                    
                    # Execute command
                    command_executed = self.execute_command(smoothed_gesture)
                    
                    # Draw landmarks
                    self.draw_landmarks(frame, landmarks_pos)
                    
                    # Draw gesture display
                    self.draw_gesture_display(frame, smoothed_gesture)
                else:
                    # No hand detected
                    # Log every 60 frames to avoid log spam
                    if self.frame_count % 60 == 0:
                        logger.info(f"Frame {self.frame_count}: [FAIL] NOT DETECTED")
                    
                    self.draw_gesture_display(frame, GestureType.UNDEFINED)
                    cv2.putText(frame, "No hand detected", (30, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Display frame
                cv2.imshow("Hand Gesture Recognition - MediaPipe", frame)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit (pressed 'q')")
                    break
                elif key == ord('s'):
                    self.print_statistics()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
        finally:
            self.shutdown()
    
    def print_statistics(self):
        """Print system statistics."""
        logger.info("")
        logger.info("="*70)
        logger.info("SYSTEM STATISTICS")
        logger.info("="*70)
        
        detection_rate = 100 * self.hands_detected / max(1, self.frame_count)
        logger.info("")
        logger.info("Frame Processing:")
        logger.info(f"  Total frames: {self.frame_count}")
        logger.info(f"  Hands detected: {self.hands_detected}")
        logger.info(f"  Detection rate: {detection_rate:.1f}%")
        
        logger.info("")
        logger.info("Gesture Classification:")
        for gesture, count in self.gesture_counts.items():
            if self.hands_detected > 0:
                pct = 100 * count / self.hands_detected
                logger.info(f"  {gesture:20s}: {count:4d} ({pct:5.1f}%)")
        
        logger.info("")
        logger.info("Command Execution (Keyboard Actions):")
        total_commands = sum(self.command_counts.values())
        for action, count in self.command_counts.items():
            if total_commands > 0:
                pct = 100 * count / total_commands
                logger.info(f"  {action:15s}: {count:4d} ({pct:5.1f}%)")
        
        logger.info("="*70)
        logger.info("")
    
    def shutdown(self):
        """Cleanup resources."""
        logger.info("")
        logger.info("="*70)
        logger.info("SHUTDOWN")
        logger.info("="*70)
        logger.info("Closing resources...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        logger.info("✓ Video capture released")
        logger.info("✓ OpenCV windows closed")
        logger.info("✓ MediaPipe detector closed")
        logger.info("")
        logger.info("Program terminated successfully.")
        logger.info("Log file saved to: " + log_filename)
        logger.info("="*70)


if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.run()
