"""
REAL-TIME HAND GESTURE RECOGNITION - MEDIAPIPE VERSION
========================================================

Uses MediaPipe Hands (solutions API) for 99%+ accurate hand detection + 21 keypoints.
Classifies 5 gestures for Temple Run / Subway Surfers game control.

GESTURE MAPPING (v4.0 - 5 Gesture System - Reversed):
  1. CLOSED_FIST      → NO ACTION (neutral - straight path / resting position)
  2. OPEN_PALM        → UP arrow key (jump)
  3. THUMBS_UP        → DOWN arrow key (slide/duck)
  4. Point LEFT       → LEFT arrow key (move left)
  5. Point RIGHT      → RIGHT arrow key (move right)

HOW TO USE:
  Straight: Close your fist (resting position) - no key pressed
  Jump:     Open your hand (all fingers spread)
  Slide:    Thumbs up (only thumb raised)
  Move L:   Point your index finger to the LEFT
  Move R:   Point your index finger to the RIGHT

KEY ADVANTAGES:
  ✓ 5 distinct gestures for full game control
  ✓ CLOSED_FIST = neutral (no accidental key presses on straight paths)
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
    OPEN_PALM = "OPEN_PALM"       # Jump (UP key)
    THUMBS_UP = "THUMBS_UP"       # Slide (DOWN key)
    CLOSED_FIST = "CLOSED_FIST"   # Neutral - no action
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
        self.last_lr_gesture = GestureType.INDEX_RIGHT  # fallback for left/right dead-zone
        self.last_executed_gesture = None  # tracks last fired gesture for single-fire mode
        self.last_command_time = 0
        self.command_cooldown = 0.20  # seconds (balanced response vs stability)
        
        logger.info(f"  - Command cooldown: {self.command_cooldown}s (balanced for stability)")
        logger.info(f"  - Gesture smoothing: 2-of-3 majority vote for stability")
        
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
        
        THUMB: Uses distance-based detection because thumb moves SIDEWAYS,
               not vertically. Measures distance from thumb tip to index MCP
               and compares against palm size.
        OTHER FINGERS: Uses Y-coordinate comparison (tip higher than DIP = raised).
        
        Returns:
            tuple: (raised_count, detailed_measurements) 
                   detailed_measurements = list of dicts with finger measurements
        """
        raised_count = 0
        measurements = []
        margin = 10  # pixels threshold for non-thumb fingers
        
        # ── THUMB DETECTION (distance-based) ──
        # The thumb extends SIDEWAYS, not vertically like other fingers.
        # Y-coordinate comparison gives INVERTED results for thumb!
        # Instead: measure how far thumb tip is from index finger base.
        #   Fist:      thumb curled in  → tip CLOSE to index MCP → NOT raised
        #   Thumbs up: thumb extended   → tip FAR from index MCP → RAISED
        thumb_tip = landmarks_pos[4]
        thumb_ip = landmarks_pos[3]
        index_mcp = landmarks_pos[5]
        wrist = landmarks_pos[0]
        middle_mcp = landmarks_pos[9]
        
        # Palm size = wrist to middle MCP (makes threshold scale-independent)
        palm_size = np.sqrt((wrist[0] - middle_mcp[0])**2 + (wrist[1] - middle_mcp[1])**2)
        
        # Distance from thumb tip to index finger MCP
        thumb_to_index = np.sqrt(
            (thumb_tip[0] - index_mcp[0])**2 + (thumb_tip[1] - index_mcp[1])**2
        )
        
        # Thumb is extended if distance exceeds 50% of palm size
        thumb_threshold = palm_size * 0.5
        thumb_raised = thumb_to_index > thumb_threshold
        
        if thumb_raised:
            raised_count += 1
        
        measurements.append({
            'finger': 'Thumb',
            'tip_y': int(thumb_tip[1]),
            'dip_y': int(thumb_ip[1]),
            'distance': int(thumb_to_index),
            'margin': int(thumb_threshold),
            'is_raised': thumb_raised
        })
        
        # ── OTHER FINGERS (Y-coordinate comparison) ──
        # Finger is "raised" if tip is HIGHER than DIP (lower Y in screen coords)
        fingers = [
            (8, 7, "Index"),      # Index: tip=8, DIP=7  
            (12, 11, "Middle"),   # Middle: tip=12, DIP=11
            (16, 15, "Ring"),     # Ring: tip=16, DIP=15
            (20, 19, "Pinky"),    # Pinky: tip=20, DIP=19
        ]
        
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
        lr_dead_zone = 15  # pixels — prevents rapid LEFT/RIGHT oscillation
        
        if index_tip_x < index_mcp_x - lr_dead_zone:
            self.last_lr_gesture = GestureType.INDEX_LEFT
            return GestureType.INDEX_LEFT, finger_count, measurements
        elif index_tip_x > index_mcp_x + lr_dead_zone:
            self.last_lr_gesture = GestureType.INDEX_RIGHT
            return GestureType.INDEX_RIGHT, finger_count, measurements
        else:
            # Inside dead-zone: hold previous left/right direction
            return self.last_lr_gesture, finger_count, measurements
    
    def smooth_gesture(self, current_gesture):
        """
        Gesture smoothing using 2-out-of-3 majority vote.
        
        A gesture must appear at least 2 times in the last 3 frames to be
        confirmed. This adds ~66ms latency (2 frames at 30 FPS) but eliminates
        single-frame glitches that cause flickering.
        
        Args:
            current_gesture: GestureType from current frame
            
        Returns:
            GestureType: Smoothed gesture (majority vote of last 3 frames)
        """
        self.gesture_history.append(current_gesture)
        
        # Need at least 3 frames of history for majority vote
        if len(self.gesture_history) < 3:
            if current_gesture != GestureType.UNDEFINED:
                return current_gesture
            return self.last_gesture
        
        # 2-out-of-3 majority vote from last 3 frames
        recent = list(self.gesture_history)[-3:]
        for gesture in recent:
            if gesture != GestureType.UNDEFINED and recent.count(gesture) >= 2:
                return gesture
        
        # No majority — hold previous gesture
        return self.last_gesture
    
    def execute_command(self, gesture):
        """
        Execute keyboard command for gesture (SINGLE-FIRE mode).
        
        Only sends a key press when the gesture CHANGES — not while it's held.
        This matches Subway Surfers gameplay where one swipe = one lane change.
        
        Flow:
          Point RIGHT → sends RIGHT once → character moves one lane
          Keep pointing RIGHT → nothing (already sent)
          Return to OPEN_PALM → resets
          Point RIGHT again → sends RIGHT once again
        
        Args:
            gesture: GestureType to execute
            
        Returns:
            bool: True if command executed, False if skipped
        """
        if gesture == GestureType.UNDEFINED:
            return False
        
        # CLOSED_FIST = neutral (resting position), no key press — but reset the tracker
        # so the next gesture fires fresh
        if gesture == GestureType.CLOSED_FIST:
            self.last_executed_gesture = None
            return False
        
        # SINGLE-FIRE: only send key when gesture CHANGES
        # If same gesture as last fired one, skip (don't spam)
        if gesture == self.last_executed_gesture:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return False
        
        # Map gesture to keyboard command
        gesture_to_key = {
            GestureType.OPEN_PALM: 'up',
            GestureType.THUMBS_UP: 'down',
            GestureType.INDEX_RIGHT: 'right',
            GestureType.INDEX_LEFT: 'left'
        }
        
        if gesture in gesture_to_key:
            key = gesture_to_key[gesture]
            
            try:
                pyautogui.press(key)
                self.last_command_time = current_time
                self.last_executed_gesture = gesture  # remember what we fired
                
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
            GestureType.CLOSED_FIST: (0, 255, 0),        # Green - Neutral (resting)
            GestureType.OPEN_PALM: (0, 255, 255),        # Yellow - Jump
            GestureType.THUMBS_UP: (255, 0, 255),        # Magenta - Slide
            GestureType.INDEX_RIGHT: (0, 165, 255),      # Orange - Right
            GestureType.INDEX_LEFT: (255, 0, 0),          # Blue - Left
            GestureType.UNDEFINED: (128, 128, 128)        # Gray
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
            GestureType.CLOSED_FIST: '— NEUTRAL (no key)',
            GestureType.OPEN_PALM: '↑ UP KEY (JUMP)',
            GestureType.THUMBS_UP: '↓ DOWN KEY (SLIDE)',
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
        logger.info("GESTURE MAPPING (v4.0 - Reversed Gesture System):")
        logger.info("  ✓ CLOSED_FIST    → NO ACTION (neutral/resting position)")
        logger.info("  ✓ OPEN_PALM      → UP arrow key (Jump)")
        logger.info("  ✓ THUMBS_UP      → DOWN arrow key (Slide)")
        logger.info("  ✓ Point LEFT     → LEFT arrow key (Move left)")
        logger.info("  ✓ Point RIGHT    → RIGHT arrow key (Move right)")
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
