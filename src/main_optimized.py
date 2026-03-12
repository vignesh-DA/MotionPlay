#!/usr/bin/env python3
"""
REAL-TIME HAND GESTURE CONTROLLER FOR TEMPLE RUN / SUBWAY SURFERS
==================================================================

GESTURE-TO-ACTION MAPPING:
  1. INDEX FINGER pointing RIGHT → RIGHT arrow key
  2. INDEX FINGER pointing LEFT  → LEFT arrow key
  3. OPEN PALM (5 fingers)       → UP arrow key (Jump)
  4. CLOSED FIST (0 fingers)     → DOWN arrow key (Slide)

VIDEO INPUT:
  - Webcam capture (1280x720, horizontal flip for mirror mode)
  - HSV skin color detection
  - Real-time contour and convex hull analysis

GESTURE DETECTION:
  - Convexity defects angle-based finger counting
  - Fingertip position for LEFT/RIGHT detection
  - Stability check: 3 consecutive frames before trigger

KEY PRESS:
  - pyautogui for keyboard simulation
  - 0.3s cooldown between gestures
  - Safe exit with 'q' key

Author: Optimized Capstone Implementation
Date: 2025
"""

import cv2
import numpy as np
import pyautogui
import time
from collections import deque
from math import sqrt, acos, degrees

# ============================================================================
# CONFIGURATION
# ============================================================================

WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
FRAME_SKIP = 1  # Process every Nth frame for speed

# HSV Skin Detection Range (optimized for your camera)
HSV_LOWER = (0, 48, 80)      # Hue, Saturation, Value lower bounds
HSV_UPPER = (20, 255, 255)   # Upper bounds

# Hand Detection Parameters
MIN_HAND_AREA = 2000         # Minimum contour area to consider as hand
MAX_HAND_AREA = 350000       # Maximum reasonable hand size

# Gesture Detection Parameters
MIN_FINGERS_THRESHOLD = 0.5  # Minimum finger length as % of hand height
DEFECT_ANGLE_THRESHOLD = 90  # Angle threshold for convexity defects (degrees)

# Gesture Stability & Cooldown
FRAME_CONFIDENCE_NEEDED = 3  # Number of frames to confirm gesture
GESTURE_COOLDOWN = 0.3       # Seconds between key presses

# Display Parameters
CONFIDENCE_COLOR = (0, 255, 0)    # Green
DETECTION_COLOR = (255, 255, 0)   # Cyan
ERROR_COLOR = (0, 0, 255)         # Red
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# ============================================================================
# HAND GESTURE DETECTOR CLASS
# ============================================================================

class HandGestureDetector:
    """
    Detects hand gestures using OpenCV contour analysis and convexity defects.
    """
    
    def __init__(self):
        self.last_gesture = "NONE"
        self.gesture_history = deque(maxlen=FRAME_CONFIDENCE_NEEDED)
        self.last_key_time = 0
        
        # Gesture constants
        self.FIST = "FIST"               # 0 fingers
        self.INDEX_LEFT = "INDEX_LEFT"   # 1 finger, left side
        self.INDEX_RIGHT = "INDEX_RIGHT" # 1 finger, right side
        self.PALM = "PALM"               # 5 fingers
        self.UNDEFINED = "UNDEFINED"
    
    def get_frame_center(self, frame):
        """Return (center_x, center_y) of frame."""
        h, w = frame.shape[:2]
        return (w // 2, h // 2)
    
    def detect_hand_contour(self, frame_hsv, frame_bgr):
        """
        Detect hand using HSV thresholding and contour analysis.
        
        Returns:
            contour: Largest hand contour, or None if not found
            frame_bgr: Original frame (for drawing)
        """
        # 1. Create HSV mask
        mask = cv2.inRange(frame_hsv, HSV_LOWER, HSV_UPPER)
        
        # 2. Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 4. Find largest contour as hand
        hand_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)
        
        if area < MIN_HAND_AREA or area > MAX_HAND_AREA:
            return None
        
        return hand_contour
    
    def get_convexity_defects(self, contour):
        """
        Extract convexity defects from hand contour.
        
        Returns:
            defects: List of defect points (fingertip valleys)
        """
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) < 3:
            return []
        
        try:
            defects = cv2.convexityDefects(contour, hull)
        except cv2.error:
            return []
        
        if defects is None:
            return []
        
        return defects
    
    def count_fingers(self, contour, defects):
        """
        Count raised fingers using convexity defects angle analysis.
        
        Algorithm:
          - For each defect, calculate angle at valley (between two finger peaks)
          - Angles < 90° indicate actual finger valleys (keep)
          - Angles > 90° indicate palm/noise (discard)
          - Number of valleys + 1 ≈ number of raised fingers
        
        Returns:
            finger_count: Number of raised fingers (0-5)
        """
        if defects is None or len(defects) < 1:
            return 0
        
        fingers = 0
        
        for defect in defects:
            s, e, f, d = defect[0]  # Start, End, Farthest point, Distance
            
            # Get coordinates
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate angle using cosine rule
            a = sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            if b == 0 or c == 0:
                continue
            
            # Clamp to avoid acos domain error
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_angle = max(-1, min(1, cos_angle))
            angle = degrees(acos(cos_angle))
            
            # Count defects with sharp angles (< 90°) = finger valleys
            if angle < DEFECT_ANGLE_THRESHOLD:
                fingers += 1
        
        return fingers
    
    def get_fingertip_position(self, contour, defects):
        """
        Get the position of the primary fingertip (usually index finger).
        
        Returns:
            fingertip_x: X coordinate of highest fingertip, or None
        """
        if not contour.size or defects is None:
            return None
        
        # Get topmost point of contour (likely fingertip)
        topmost_point = tuple(contour[contour[:, :, 1].argmin()][0])
        return topmost_point[0]
    
    def classify_gesture(self, finger_count, fingertip_x, frame_width):
        """
        Map finger count and position to gesture class.
        
        Logic:
          - finger_count == 0 → FIST
          - finger_count == 1 AND fingertip_x < frame_width/2 → INDEX_LEFT
          - finger_count == 1 AND fingertip_x >= frame_width/2 → INDEX_RIGHT
          - finger_count == 5 → PALM
          - else → UNDEFINED
        
        Args:
            finger_count: Number of fingers detected
            fingertip_x: X position of fingertip
            frame_width: Width of video frame
        
        Returns:
            gesture: Gesture string constant
        """
        if finger_count == 0:
            return self.FIST
        elif finger_count == 1:
            center_x = frame_width // 2
            if fingertip_x is None:
                return self.UNDEFINED
            return self.INDEX_LEFT if fingertip_x < center_x else self.INDEX_RIGHT
        elif finger_count == 5:
            return self.PALM
        else:
            return self.UNDEFINED
    
    def detect(self, frame_bgr, frame_hsv):
        """
        Full gesture detection pipeline.
        
        Returns:
            gesture: Detected gesture string
            contour: Hand contour for drawing
            defects: Convexity defects for drawing
            finger_count: Number of fingers
        """
        h, w = frame_bgr.shape[:2]
        
        # Step 1: Detect hand
        contour = self.detect_hand_contour(frame_hsv, frame_bgr)
        
        if contour is None:
            self.gesture_history.append(self.UNDEFINED)
            return self.UNDEFINED, None, None, 0
        
        # Step 2: Get convexity defects
        defects = self.get_convexity_defects(contour)
        
        # Step 3: Count fingers
        finger_count = self.count_fingers(contour, defects)
        
        # Step 4: Get fingertip position
        fingertip_x = self.get_fingertip_position(contour, defects)
        
        # Step 5: Classify gesture
        gesture = self.classify_gesture(finger_count, fingertip_x, w)
        
        # Step 6: Stability check (need consistent gesture for N frames)
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) == FRAME_CONFIDENCE_NEEDED:
            # Check if all recent frames agree
            if all(g == gesture for g in self.gesture_history):
                self.last_gesture = gesture
        
        return gesture, contour, defects, finger_count


# ============================================================================
# GAME CONTROL CLASS
# ============================================================================

class GameController:
    """
    Controls game via keyboard simulation.
    Maps gestures to game actions.
    """
    
    def __init__(self):
        self.last_key_time = 0
        self.gesture_to_key = {
            "FIST": "down",           # DOWN arrow - slide
            "INDEX_LEFT": "left",     # LEFT arrow - move left
            "INDEX_RIGHT": "right",   # RIGHT arrow - move right
            "PALM": "up",             # UP arrow - jump
        }
    
    def send_key(self, gesture):
        """
        Send keyboard command based on gesture, with cooldown check.
        
        Args:
            gesture: Gesture string
        
        Returns:
            True if key was sent, False if in cooldown
        """
        if gesture not in self.gesture_to_key:
            return False
        
        current_time = time.time()
        if current_time - self.last_key_time < GESTURE_COOLDOWN:
            return False
        
        key = self.gesture_to_key[gesture]
        pyautogui.press(key)
        self.last_key_time = current_time
        
        return True


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HandGestureGameApp:
    """
    Main application combining hand detection + game control.
    """
    
    def __init__(self):
        self.detector = HandGestureDetector()
        self.controller = GameController()
        self.frame_count = 0
        self.fps_time = time.time()
        self.fps_counter = 0
    
    def draw_debug_info(self, frame, gesture, finger_count, contour, defects):
        """
        Draw gesture info and hand analysis on frame.
        """
        h, w = frame.shape[:2]
        
        # Gesture label (top-left)
        key = self.controller.gesture_to_key.get(gesture, "NONE")
        label = f"Gesture: {gesture} | Key: {key.upper()}"
        color = CONFIDENCE_COLOR if gesture != "UNDEFINED" else ERROR_COLOR
        cv2.putText(frame, label, (10, 30), TEXT_FONT, TEXT_SCALE, 
                    color, TEXT_THICKNESS)
        
        # Finger count
        cv2.putText(frame, f"Fingers: {finger_count}", (10, 65), 
                    TEXT_FONT, TEXT_SCALE, DETECTION_COLOR, TEXT_THICKNESS)
        
        # FPS
        self.fps_counter += 1
        if time.time() - self.fps_time > 1.0:
            self.fps_display = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
        
        fps_text = f"FPS: {getattr(self, 'fps_display', 0)}"
        cv2.putText(frame, fps_text, (10, 100), TEXT_FONT, TEXT_SCALE, 
                    DETECTION_COLOR, TEXT_THICKNESS)
        
        # Draw hand contour and hull
        if contour is not None:
            cv2.drawContours(frame, [contour], 0, DETECTION_COLOR, 2)
            
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], 0, CONFIDENCE_COLOR, 2)
            
            # Draw fingertips (from convex hull corners)
            for point in hull:
                cv2.circle(frame, tuple(point[0]), 5, (0, 255, 255), -1)
    
    def run(self):
        """
        Main application loop.
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return
        
        print("[OK] Webcam opened. Press 'q' to quit.")
        print("[OK] System ready - perform gestures in front of camera")
        
        frame_index = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[WARNING] Failed to read frame")
                    break
                
                frame_index += 1
                
                # Skip frames for performance
                if frame_index % FRAME_SKIP != 0:
                    continue
                
                # Mirror for natural viewing
                frame = cv2.flip(frame, 1)
                
                # Convert to HSV
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Detect gesture
                gesture, contour, defects, finger_count = self.detector.detect(
                    frame, frame_hsv
                )
                
                # Send key if gesture is stable
                if gesture != "UNDEFINED":
                    self.controller.send_key(gesture)
                
                # Draw debug info
                self.draw_debug_info(frame, gesture, finger_count, contour, defects)
                
                # Display
                cv2.imshow("Hand Gesture Controller - Temple Run / Subway Surfers", frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[OK] Shutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\n[OK] Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[OK] Application closed")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = HandGestureGameApp()
    app.run()
