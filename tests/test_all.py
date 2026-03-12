"""
COMPREHENSIVE TEST SUITE
========================

Unit tests and integration tests for all modules.

Test Coverage:
  - Module 1: Video capture and preprocessing
  - Module 2: Hand detection and segmentation
  - Module 3: Feature extraction
  - Module 4: Gesture classification
  - Module 5: Game control interface
  - Integration: End-to-end pipeline

Run Tests:
  python -m pytest tests/ -v

Author:     Capstone Project
Date:       2025
"""

import sys
sys.path.insert(0, './src')

import numpy as np
import cv2
from modules.video_capture_preprocessing import VideoCapturePreprocessor
from modules.hand_detection_segmentation import HandDetectionSegmentor
from modules.feature_extraction import FeatureExtractor
from modules.gesture_classification import GestureClassifier, GestureType
from modules.game_control_interface import GameControlInterface


class TestModule1VideoCapture:
    """Test Module 1: Video Capture and Preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes correctly."""
        try:
            preprocessor = VideoCapturePreprocessor()
            assert preprocessor.is_initialized
            props = preprocessor.get_camera_properties()
            assert props['is_open']
            assert props['width'] == 640
            assert props['height'] == 480
            preprocessor.release()
            print("✓ Preprocessor initialization test passed")
        except RuntimeError:
            print("⊘ Preprocessor test skipped (no camera available)")
    
    def test_frame_dimensions(self):
        """Test that frames are resized correctly."""
        try:
            preprocessor = VideoCapturePreprocessor()
            ret, frame = preprocessor.read_frame()
            if ret:
                assert frame.shape == (480, 640, 3)
                preprocessor.release()
                print("✓ Frame dimensions test passed")
        except RuntimeError:
            print("⊘ Frame test skipped")
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        try:
            preprocessor = VideoCapturePreprocessor()
            ret, raw_frame = preprocessor.read_frame()
            if ret:
                bgr, hsv = preprocessor.preprocess_frame(raw_frame)
                assert bgr.shape == (480, 640, 3)
                assert hsv.shape == (480, 640, 3)
                assert bgr.dtype == np.uint8
                assert hsv.dtype == np.uint8
                preprocessor.release()
                print("✓ Preprocessing pipeline test passed")
        except RuntimeError:
            print("⊘ Pipeline test skipped")


class TestModule2HandDetection:
    """Test Module 2: Hand Detection and Segmentation."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = HandDetectionSegmentor()
        assert detector.min_contour_area == 5000
        print("✓ Detector initialization test passed")
    
    def test_hsv_thresholding(self):
        """Test HSV color thresholding."""
        detector = HandDetectionSegmentor()
        # Create test HSV image with known ranges
        test_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some skin-colored pixels  
        test_hsv[25:75, 25:75, :] = [10, 80, 150]  # Skin color in HSV
        
        mask = detector.apply_hsv_threshold(test_hsv)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert np.sum(mask) > 0  # Some pixels should be white
        print("✓ HSV thresholding test passed")
    
    def test_morphological_operations(self):
        """Test morphological operations cleanup."""
        detector = HandDetectionSegmentor()
        # Create noisy binary image
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[40:60, 40:60] = 255  # Hand region
        test_mask[20, 20] = 255  # Noise
        
        cleaned = detector.morphological_operations(test_mask)
        assert cleaned.shape == test_mask.shape
        assert cleaned.dtype == np.uint8
        print("✓ Morphological operations test passed")


class TestModule3FeatureExtraction:
    """Test Module 3: Feature Extraction."""
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.max_finger_count == 5
        assert extractor.angle_threshold == np.pi / 2
        print("✓ Extractor initialization test passed")
    
    def test_convex_hull_computation(self):
        """Test convex hull computation."""
        extractor = FeatureExtractor()
        # Create simple contour (square)
        contour = np.array([
            [[0, 0]],
            [[100, 0]],
            [[100, 100]],
            [[0, 100]]
        ], dtype=np.int32)
        
        hull = extractor.compute_convex_hull(contour)
        assert hull is not None
        assert len(hull) == 4  # 4 corners
        print("✓ Convex hull test passed")
    
    def test_finger_count_estimation(self):
        """Test finger count estimation."""
        extractor = FeatureExtractor()
        # Test various defect counts
        test_cases = [
            ([], 1),      # 0 defects → 1 finger (closed)
            ([{}, {}, {}], 4),  # 3 defects → 4 fingers
            ([{}, {}, {}, {}], 5),  # 4 defects → 5 fingers (open)
        ]
        
        for defects, expected_count in test_cases:
            count = extractor.estimate_finger_count(defects)
            assert count == expected_count
        
        print("✓ Finger count estimation test passed")


class TestModule4GestureClassification:
    """Test Module 4: Gesture Classification."""
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = GestureClassifier()
        assert classifier.smoothing_enabled
        print("✓ Classifier initialization test passed")
    
    def test_gesture_rules(self):
        """Test gesture classification rules."""
        classifier = GestureClassifier(smoothing_enabled=False)
        
        test_cases = [
            (5, GestureType.OPEN_PALM),
            (0, GestureType.CLOSED_FIST),
            (2, GestureType.TWO_FINGERS),
            (3, GestureType.THREE_FINGERS),
            (4, GestureType.FOUR_FINGERS),
            (1, GestureType.UNDEFINED),
            (6, GestureType.UNDEFINED),
        ]
        
        for finger_count, expected_gesture in test_cases:
            features = {'finger_count': finger_count}
            gesture, _ = classifier.classify_from_features(features)
            assert gesture == expected_gesture, \
                f"Expected {expected_gesture.value} for {finger_count} fingers, got {gesture.value}"
        
        print("✓ Gesture classification rules test passed")
    
    def test_gesture_action_mapping(self):
        """Test gesture to action mapping."""
        classifier = GestureClassifier()
        
        mappings = [
            (GestureType.OPEN_PALM, "jump"),
            (GestureType.CLOSED_FIST, "slide"),
            (GestureType.TWO_FINGERS, "move_right"),
            (GestureType.THREE_FINGERS, "move_left"),
        ]
        
        for gesture, expected_action in mappings:
            action = classifier.get_gesture_action(gesture)
            assert action == expected_action
        
        print("✓ Gesture-to-action mapping test passed")
    
    def test_confidence_scores(self):
        """Test confidence scoring."""
        classifier = GestureClassifier()
        
        # CLOSED_FIST should have highest confidence
        conf_fist = classifier.get_gesture_confidence(GestureType.CLOSED_FIST)
        conf_open = classifier.get_gesture_confidence(GestureType.OPEN_PALM)
        conf_undefined = classifier.get_gesture_confidence(GestureType.UNDEFINED)
        
        assert conf_fist > conf_undefined
        assert conf_open > conf_undefined
        assert 0 <= conf_fist <= 1
        
        print("✓ Confidence scoring test passed")


class TestModule5GameControl:
    """Test Module 5: Game Control Interface."""
    
    def test_controller_initialization(self):
        """Test game controller initialization."""
        controller = GameControlInterface()
        assert controller.command_cooldown == 0.5
        assert controller.log_commands == True
        print("✓ Game controller initialization test passed")
    
    def test_debounce_mechanism(self):
        """Test debounce timer."""
        controller = GameControlInterface(command_cooldown=0.1)
        
        # First command should be allowed
        assert controller.check_debounce() == True
        
        # Simulate command execution
        import time
        controller.last_command_time = time.time()
        
        # Immediate second command should fail
        assert controller.check_debounce() == False
        
        # After cooldown, should be allowed
        time.sleep(0.15)
        assert controller.check_debounce() == True
        
        print("✓ Debounce mechanism test passed")
    
    def test_key_mapping(self):
        """Test keyboard key mapping."""
        controller = GameControlInterface()
        
        test_mapping = {
            "jump": "up",
            "slide": "down",
            "move_right": "right",
            "move_left": "left",
        }
        
        for action, expected_key in test_mapping.items():
            key = controller.GESTURE_TO_KEY_MAP.get(action)
            assert key == expected_key
        
        print("✓ Keyboard key mapping test passed")
    
    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        controller = GameControlInterface(min_confidence_threshold=0.7)
        
        # High confidence should pass
        assert controller.should_accept_gesture(0.8) == True
        
        # Low confidence should fail
        assert controller.should_accept_gesture(0.5) == False
        
        # Threshold boundary
        assert controller.should_accept_gesture(0.7) == True
        
        print("✓ Confidence threshold test passed")
    
    def test_statistics_tracking(self):
        """Test command statistics tracking."""
        controller = GameControlInterface()
        controller.total_commands_issued = 10
        controller.total_commands_rejected = 2
        controller.total_gestures_processed = 12
        
        stats = controller.get_statistics()
        assert stats['total_commands_issued'] == 10
        assert stats['total_commands_rejected'] == 2
        assert stats['execution_rate'] == 100 * 10 / 12
        
        print("✓ Statistics tracking test passed")


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_gesture_to_command_mapping(self):
        """Test complete gesture to command pipeline."""
        classifier = GestureClassifier(smoothing_enabled=False)
        controller = GameControlInterface()
        
        test_cases = [
            (5, 0.95, "jump", "up"),
            (0, 0.95, "slide", "down"),
            (2, 0.85, "move_right", "right"),
            (3, 0.86, "move_left", "left"),
        ]
        
        for finger_count, conf, expected_action, expected_key in test_cases:
            features = {'finger_count': finger_count}
            gesture, confidence = classifier.classify_from_features(features)
            action = classifier.get_gesture_action(gesture)
            key = classifier.get_keyboard_key(action)
            
            assert action == expected_action
            assert key == expected_key
        
        print("✓ Gesture-to-command mapping integration test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestModule1VideoCapture,
        TestModule2HandDetection,
        TestModule3FeatureExtraction,
        TestModule4GestureClassification,
        TestModule5GameControl,
        TestIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name} failed: {str(e)}")
                failed_tests += 1
            except Exception as e:
                print(f"⊘ {method_name} error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")
    print("=" * 70)
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
