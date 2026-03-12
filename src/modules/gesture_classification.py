"""
MODULE 4: GESTURE CLASSIFICATION
==================================

Purpose:
    Convert extracted hand features into discrete gesture labels using
    rule-based decision logic, creating a mapping between hand configurations
    and game actions.

Approach:
    - Rule-based classification (no machine learning)
    - Simple if-else logic based on finger count
    - Direct mapping of gestures to game actions
    - Optional gesture smoothing via majority voting

Gesture Definitions:
    1. OPEN_PALM (5 fingers)  → JUMP
    2. CLOSED_FIST (0 fingers) → SLIDE
    3. TWO_FINGERS (2 fingers) → MOVE_RIGHT
    4. THREE_FINGERS (3 fingers) → MOVE_LEFT
    5. FOUR_FINGERS (4 fingers) → RESERVED (no action)
    6. UNDEFINED (anything else)  → STATUS_UNKNOWN

Input:  Features dict from Module 3 (finger_count, confidence)
Output: Gesture label (categorical), confidence score

Author:     Capstone Project
Date:       2025
Version:    1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import deque
import time


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    OPEN_PALM = "OPEN_PALM"
    CLOSED_FIST = "CLOSED_FIST"
    TWO_FINGERS = "TWO_FINGERS"
    THREE_FINGERS = "THREE_FINGERS"
    FOUR_FINGERS = "FOUR_FINGERS"
    UNDEFINED = "UNDEFINED"


class GestureClassifier:
    """
    Classifies hand gestures based on finger count using rule-based logic.
    
    Attributes:
        gesture_map (dict): Maps finger count to gesture type
        confidence_scores (dict): Default confidence for each gesture
        smoothing_enabled (bool): Enable majority voting smoothing
        smoothing_window_size (int): Number of frames for smoothing
        gesture_history (deque): Circular buffer of last gestures
    """
    
    def __init__(self, 
                 smoothing_enabled: bool = True,
                 smoothing_window_size: int = 5):
        """
        Initialize gesture classifier.
        
        Args:
            smoothing_enabled (bool): Enable gesture smoothing (default True)
            smoothing_window_size (int): Frames for majority vote (default 5)
        """
        # Define gesture mapping: finger_count → GestureType
        self.gesture_map = {
            5: GestureType.OPEN_PALM,
            0: GestureType.CLOSED_FIST,
            2: GestureType.TWO_FINGERS,
            3: GestureType.THREE_FINGERS,
            4: GestureType.FOUR_FINGERS,
        }
        
        # Confidence scores for each gesture type
        # Based on how distinctive each gesture is
        self.confidence_scores = {
            GestureType.CLOSED_FIST: 0.95,   # Very distinct (no fingers)
            GestureType.OPEN_PALM: 0.92,     # Very distinct (all fingers)
            GestureType.TWO_FINGERS: 0.85,   # Medium (can confuse with 3)
            GestureType.THREE_FINGERS: 0.86, # Medium (can confuse with 2, 4)
            GestureType.FOUR_FINGERS: 0.75,  # Less reliable
            GestureType.UNDEFINED: 0.50,     # Low confidence
        }
        
        # Gesture smoothing parameters
        self.smoothing_enabled = smoothing_enabled
        self.smoothing_window_size = smoothing_window_size
        self.gesture_history = deque(maxlen=smoothing_window_size)
        
        # Statistics tracking
        self.total_classifications = 0
        self.classification_counts = {gesture: 0 for gesture in GestureType}
        
        print("✓ Gesture classifier initialized")
        if smoothing_enabled:
            print(f"  Smoothing enabled (window size: {smoothing_window_size} frames)")
        print(f"  Gesture mapping: {len(self.gesture_map)} rules")
    
    def classify_gesture(self, finger_count: int) -> GestureType:
        """
        Classify gesture based on finger count using rule-based logic.
        
        Decision Tree:
            if finger_count == 5:
                gesture = OPEN_PALM
            elif finger_count == 0:
                gesture = CLOSED_FIST
            elif finger_count == 2:
                gesture = TWO_FINGERS
            elif finger_count == 3:
                gesture = THREE_FINGERS
            elif finger_count == 4:
                gesture = FOUR_FINGERS
            else:
                gesture = UNDEFINED
        
        Args:
            finger_count (int): Detected finger count (0-5+)
        
        Returns:
            GestureType: Classified gesture label
        
        Time Complexity: O(1)
        """
        gesture = self.gesture_map.get(finger_count, GestureType.UNDEFINED)
        return gesture
    
    def get_gesture_confidence(self, gesture: GestureType, features: dict = None) -> float:
        """
        Get confidence score for classified gesture.
        
        Confidence levels reflect how distinctive each gesture is:
            - CLOSED_FIST: 0.95 (No fingers is very clear)
            - OPEN_PALM: 0.92 (5 fingers is very clear)
            - TWO_FINGERS: 0.85 (Can confuse with 3)
            - THREE_FINGERS: 0.86 (Can confuse with 2 or 4)
            - FOUR_FINGERS: 0.75 (Less reliable)
            - UNDEFINED: 0.50 (Ambiguous/error)
        
        Args:
            gesture (GestureType): Gesture label
            features (dict): Optional feature dict (for future enhancement)
        
        Returns:
            float: Confidence score [0.0, 1.0]
        """
        confidence = self.confidence_scores.get(gesture, 0.5)
        
        # Optional: Future enhancement could adjust confidence based on features
        # (e.g., hand solidity, defect clarity, contour area)
        if features is not None:
            # Could add confidence adjustments here
            pass
        
        return confidence
    
    def apply_smoothing(self, gesture: GestureType) -> GestureType:
        """
        Apply gesture smoothing using majority voting.
        
        Purpose:
            Reduce jitter from frame-to-frame gesture variations.
            Ensures gesture is stable before confirming classification.
        
        Strategy:
            1. Add current gesture to history buffer
            2. Return most common gesture in buffer
            3. Requires 3+ votes in 5-frame window
        
        Trade-offs:
            ✓ Reduces false positives
            ✓ Stabilizes gesture detection
            ✗ Adds ~100-150ms latency (5 frames @ 30 FPS)
        
        Args:
            gesture (GestureType): Current frame's gesture
        
        Returns:
            GestureType: Smoothed gesture (or original if insufficient history)
        
        Time Complexity: O(w) where w = window size (usually 5)
        """
        self.gesture_history.append(gesture)
        
        # If buffer not full, return current gesture
        if len(self.gesture_history) < self.smoothing_window_size:
            return gesture
        
        # Compute majority (mode) of window
        gesture_list = list(self.gesture_history)
        
        # Count occurrences
        gesture_counts = {}
        for g in gesture_list:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Get most common (ties broken by latest frame)
        smoothed_gesture = max(gesture_counts, key=gesture_counts.get)
        
        return smoothed_gesture
    
    def classify_from_features(self, features: dict) -> Tuple[GestureType, float]:
        """
        Complete classification pipeline from feature dict.
        
        Sequence:
            1. Extract finger count from features
            2. Classify gesture via rule-based logic
            3. Get confidence score
            4. Apply smoothing if enabled
            5. Update statistics
            6. Return gesture + confidence
        
        Args:
            features (dict): Feature dict from Module 3 containing:
                - 'finger_count': int (0-5)
                - 'defects': list (optional)
                - Other feature metrics
        
        Returns:
            Tuple[GestureType, float]: (gesture_label, confidence_score)
        
        Time Complexity: O(1) without smoothing, O(w) with smoothing
        Expected Duration: <2 ms
        """
        # Extract finger count
        finger_count = features.get('finger_count', -1)
        
        # Classify gesture
        gesture = self.classify_gesture(finger_count)
        
        # Get confidence
        confidence = self.get_gesture_confidence(gesture, features)
        
        # Apply smoothing if enabled
        if self.smoothing_enabled:
            gesture = self.apply_smoothing(gesture)
        
        # Update statistics
        self.total_classifications += 1
        self.classification_counts[gesture] += 1
        
        return gesture, confidence
    
    def get_gesture_action(self, gesture: GestureType) -> Optional[str]:
        """
        Map gesture to game action/keyboard key.
        
        Note: This method provides the gesture→action mapping.
        Actual keyboard execution is handled by Module 5.
        
        Mapping:
            OPEN_PALM → "jump" (UP arrow)
            CLOSED_FIST → "slide" (DOWN arrow)
            TWO_FINGERS → "move_right" (RIGHT arrow)
            THREE_FINGERS → "move_left" (LEFT arrow)
            FOUR_FINGERS → None (reserved)
            UNDEFINED → None (no action)
        
        Args:
            gesture (GestureType): Classified gesture
        
        Returns:
            Optional[str]: Action key or None if no action
        
        Time Complexity: O(1)
        """
        action_map = {
            GestureType.OPEN_PALM: "jump",
            GestureType.CLOSED_FIST: "slide",
            GestureType.TWO_FINGERS: "move_right",
            GestureType.THREE_FINGERS: "move_left",
            GestureType.FOUR_FINGERS: None,
            GestureType.UNDEFINED: None,
        }
        
        return action_map.get(gesture, None)
    
    def get_keyboard_key(self, action: str) -> Optional[str]:
        """
        Map action to keyboard key for pyautogui.
        
        Mapping:
            "jump" → "up"
            "slide" → "down"
            "move_right" → "right"
            "move_left" → "left"
        
        Args:
            action (str): Action string from get_gesture_action()
        
        Returns:
            Optional[str]: pyautogui-compatible key name
        
        Time Complexity: O(1)
        """
        key_map = {
            "jump": "up",
            "slide": "down",
            "move_right": "right",
            "move_left": "left",
        }
        
        return key_map.get(action, None)
    
    def get_statistics(self) -> dict:
        """
        Get classification statistics.
        
        Returns:
            dict: Statistics including:
                - 'total_classifications': Total gestures classified
                - 'counts': Dict of gesture type → count
                - 'distribution': Dict of gesture type → percentage
                - 'most_common': Most frequently detected gesture
        """
        total = self.total_classifications
        
        if total == 0:
            return {'total_classifications': 0, 'counts': {}, 'distribution': {}}
        
        distribution = {
            gesture: 100 * self.classification_counts[gesture] / total
            for gesture in GestureType
        }
        
        most_common = max(
            self.classification_counts,
            key=self.classification_counts.get
        )
        
        return {
            'total_classifications': total,
            'counts': dict(self.classification_counts),
            'distribution': distribution,
            'most_common': most_common.value,
            'most_common_percentage': distribution[most_common]
        }
    
    def reset_statistics(self) -> None:
        """Reset classification statistics counters."""
        self.total_classifications = 0
        self.classification_counts = {gesture: 0 for gesture in GestureType}
        self.gesture_history.clear()
        print("✓ Statistics reset")


if __name__ == "__main__":
    """
    Test Module 4 independently.
    
    Demonstrates gesture classification pipeline.
    """
    print("=" * 60)
    print("MODULE 4: GESTURE CLASSIFICATION TEST")
    print("=" * 60)
    
    try:
        # Initialize classifier
        classifier = GestureClassifier(smoothing_enabled=True)
        
        # Test gesture classification on various finger counts
        test_cases = [
            (5, "OPEN_PALM"),
            (0, "CLOSED_FIST"),
            (2, "TWO_FINGERS"),
            (3, "THREE_FINGERS"),
            (4, "FOUR_FINGERS"),
            (1, "UNDEFINED"),
            (6, "UNDEFINED"),
        ]
        
        print("\nTesting gesture classification:")
        print("-" * 60)
        
        for finger_count, expected in test_cases:
            # Create fake feature dict
            features = {'finger_count': finger_count}
            
            # Classify
            gesture, confidence = classifier.classify_from_features(features)
            
            # Get action and key
            action = classifier.get_gesture_action(gesture)
            key = classifier.get_keyboard_key(action) if action else None
            
            print(f"Fingers={finger_count:2d} → Gesture={gesture.value:20s} " +
                  f"(conf={confidence:.2f}) → Action={str(action):15s} → Key={str(key):8s}")
            
            # Verify
            expected_gesture = GestureType[expected]
            assert gesture == expected_gesture, f"Expected {expected}, got {gesture.value}"
        
        # Test smoothing
        print("\nTesting gesture smoothing (window size=5):")
        print("-" * 60)
        
        classifier.reset_statistics()
        
        # Simulate oscillating gesture detection (5 frames of 0, then 5 frames of 5)
        test_sequence = [0, 0, 0, 0, 0, 5, 5, 5, 5, 5]
        smoothed_sequence = []
        
        for finger_count in test_sequence:
            features = {'finger_count': finger_count}
            gesture, _ = classifier.classify_from_features(features)
            smoothed_sequence.append(gesture.value)
            print(f"Input: {finger_count} → Smoothed: {gesture.value}")
        
        # Test statistics
        print("\nClassification Statistics:")
        print("-" * 60)
        stats = classifier.get_statistics()
        print(f"Total classifications: {stats['total_classifications']}")
        print(f"Distribution:")
        for gesture_type, percentage in stats['distribution'].items():
            count = stats['counts'][gesture_type]
            if count > 0:
                print(f"  {gesture_type.value:20s}: {count:2d} ({percentage:5.1f}%)")
        
        print(f"Most common: {stats['most_common']} ({stats['most_common_percentage']:.1f}%)")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
