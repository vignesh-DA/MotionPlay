"""
Hand Gesture Recognition Modules Package

This package contains all 5 modules of the hand gesture recognition system:
  1. video_capture_preprocessing - Video capture and frame preprocessing
  2. hand_detection_segmentation - Hand detection and binary masking
  3. feature_extraction - Geometric feature extraction (convex hull, defects)
  4. gesture_classification - Rule-based gesture classification
  5. game_control_interface - Keyboard command execution and debouncing

Author: Capstone Project
Date: 2025
"""

__all__ = [
    'VideoCapturePreprocessor',
    'HandDetectionSegmentor',
    'FeatureExtractor',
    'GestureClassifier',
    'GameControlInterface',
]
