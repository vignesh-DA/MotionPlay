"""
MODULE 3: FEATURE EXTRACTION
=============================

Purpose:
    Extract geometric features from hand contour to characterize hand configuration,
    particularly identifying individual fingers and their positions.

Core Algorithm: Convex Hull + Convexity Defects Analysis
    - Convex hull: Smallest convex polygon enclosing hand
    - Convexity defects: Deepest valleys between fingers
    - Finger count estimation: number_of_defects + 1

Responsibilities:
    - Compute convex hull of hand contour
    - Detect convexity defects
    - Filter defects by angle (< 90°)
    - Count raised fingers
    - Identify fingertip positions

Input:  Hand contour from Module 2
Output: Finger count (0-5), fingertip coordinates, hull/defect data

Author:     Capstone Project
Date:       2025
Version:    1.0
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class FeatureExtractor:
    """
    Extracts hand features using convex hull and convexity defects analysis.
    
    Attributes:
        max_finger_count (int): Maximum fingers to detect (5)
        angle_threshold (float): Minimum angle for valid defect (90 degrees/1.57 rad)
        contour_approx_epsilon (float): Contour approximation factor (0.02)
    """
    
    def __init__(self, 
                 max_finger_count: int = 5,
                 angle_threshold: float = np.pi / 2):
        """
        Initialize feature extractor.
        
        Args:
            max_finger_count (int): Maximum fingers to detect (default 5)
            angle_threshold (float): Angle threshold in radians (default π/2 = 90°)
        """
        self.max_finger_count = max_finger_count
        self.angle_threshold = angle_threshold
        self.contour_approx_epsilon = 0.02
        
        print("✓ Feature extractor initialized")
        print(f"  Max fingers: {max_finger_count}")
        print(f"  Angle threshold: {np.degrees(angle_threshold):.1f}°")
    
    def approximate_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Approximate contour to reduce noise while preserving shape.
        
        Algorithm: Ramer-Douglas-Peucker approximation
            Simplifies contour by removing points that are too close to a line
            between their neighbors. Reduces noise while preserving corners.
        
        Args:
            contour (np.ndarray): Original hand contour
        
        Returns:
            np.ndarray: Approximated contour with fewer points
        
        Time Complexity: O(n log n)
        Expected Duration: <1 ms
        """
        epsilon = self.contour_approx_epsilon * cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        return approximated
    
    def compute_convex_hull(self, contour: np.ndarray) -> np.ndarray:
        """
        Compute convex hull of hand contour.
        
        Mathematical Definition:
            The convex hull is the smallest convex polygon enclosing all points.
        
        Algorithm: Monotone Chain (Andrew's Algorithm)
            Time Complexity: O(n log n)
            Sorts points by x-coordinate, computes lower and upper hull separately.
        
        For a hand, convex hull vertices approximately correspond to fingertips.
        
        Args:
            contour (np.ndarray): Hand contour
        
        Returns:
            np.ndarray: Convex hull vertices, shape (num_vertices, 1, 2)
        
        Time Complexity: O(n log n)
        Expected Duration: <1 ms
        """
        hull = cv2.convexHull(contour, returnPoints=True)
        return hull
    
    def compute_convexity_defects(self, 
                                 contour: np.ndarray,
                                 hull: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect convexity defects between contour and convex hull.
        
        Mathematical Definition:
            A convexity defect is a region where the actual contour is "inside"
            the convex hull. Represents the space between two fingers.
        
        Algorithm:
            1. Use cv2.convexityDefects() to compute defects
            2. For each defect: (start_idx, end_idx, farthest_idx, depth)
               - start_idx, end_idx: Contour indices where defect touches hull
               - farthest_idx: Index of deepest point of defect
               - depth: Distance from farthest point to hull edge
        
        Args:
            contour (np.ndarray): Hand contour, shape (N, 1, 2)
            hull (np.ndarray): Convex hull, shape (M, 1, 2)
        
        Returns:
            Optional[np.ndarray]: Defects array [start, end, farthest, depth]
                                 Each row is one defect
                                 None if no defects found
        
        Notes:
            - Defects must be checked for validity (depth > 0)
            - Angle filtering is applied in separate function
        
        Time Complexity: O(n)
        Expected Duration: <1 ms
        """
        try:
            # Convert hull to required format for convexityDefects
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            
            # Compute defects
            # Note: Can fail if contour has self-intersections
            defects = cv2.convexityDefects(contour, hull_indices)
            
            return defects
        except cv2.error as e:
            # Handle self-intersecting contours (common with poor hand segmentation)
            # Return None to indicate defect computation failed
            # The calling function will handle this gracefully
            return None
    
    def filter_defects_by_angle(self, 
                               contour: np.ndarray,
                               defects: np.ndarray) -> List[dict]:
        """
        Filter convexity defects by angle threshold.
        
        Mathematical Basis: Cosine Rule
            For a defect with points A (start), B (end), C (farthest):
            
            vec1 = C - A
            vec2 = C - B
            
            cos(angle) = (vec1 · vec2) / (|vec1| × |vec2|)
            angle = arccos(cos(angle))
        
        Filtering Logic:
            Only keep defects where angle < 90°
            
            - angle < 90°: Sharp valley between distinct fingers
            - angle ≥ 90°: Shallow angle indicates hand not fully open
            
        Example:
            Open hand (5 fingers): ~4 defects with sharp angles (<90°)
            Partially open: 2-3 defects meet criteria
            Closed fist: 0 defects
        
        Args:
            contour (np.ndarray): Hand contour
            defects (np.ndarray): Defects from compute_convexity_defects()
        
        Returns:
            List[dict]: Filtered defects with properties:
                - 'start': Start point (x, y)
                - 'end': End point (x, y)
                - 'farthest': Farthest point (x, y)
                - 'depth': Euclidean distance (pixels)
                - 'angle': Angle in radians
                - 'angle_deg': Angle in degrees
        
        Time Complexity: O(k) where k = number of defects (typically <10)
        Expected Duration: <1 ms
        """
        valid_defects = []
        
        if defects is None:
            return valid_defects
        
        for defect in defects:
            s, e, f, d = defect[0]
            
            # Extract points from contour
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Compute vectors
            vec1 = np.array(start) - np.array(far)
            vec2 = np.array(end) - np.array(far)
            
            # Compute angle using cosine rule
            dot_product = np.dot(vec1, vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                continue
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            # Clamp to [-1, 1] to avoid numerical errors with arccos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Filter by angle threshold
            # depth > 0 ensures it's a real defect (not noise)
            if angle < self.angle_threshold and d > 0:
                valid_defects.append({
                    'start': start,
                    'end': end,
                    'farthest': far,
                    'depth': d,
                    'angle': angle,
                    'angle_deg': np.degrees(angle)
                })
        
        return valid_defects
    
    def estimate_finger_count(self, valid_defects: List[dict]) -> int:
        """
        Estimate number of raised fingers from valid defects.
        
        Mathematical Relationship:
            finger_count = number_of_valid_defects + 1
        
        Reasoning:
            - N distinct fingers create N-1 gaps between them
            - Each gap = 1 convexity defect
            - Therefore: N fingers = (N-1) defects + 1
        
        Examples:
            - Open palm (5 fingers): 4 gaps → 4 defects → finger_count = 5
            - Two fingers (peace): 1 gap → 1 defect → finger_count = 2
            - Closed fist (0 fingers): 0 gaps → 0 defects → finger_count = 0
        
        Args:
            valid_defects (List[dict]): Filtered defects with angle < 90°
        
        Returns:
            int: Estimated finger count (0-5)
        
        Time Complexity: O(1)
        """
        # Add 1 because N defects create N+1 "finger spaces"
        finger_count = len(valid_defects) + 1
        
        # Clamp to valid range [0, max_finger_count]
        finger_count = min(finger_count, self.max_finger_count)
        finger_count = max(finger_count, 0)
        
        return finger_count
    
    def get_fingertip_positions(self, hull: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identify approximate fingertip positions from convex hull vertices.
        
        Assumption:
            Convex hull vertices roughly correspond to fingertips for
            an open hand configuration. Works best when hand is well-separated.
        
        Args:
            hull (np.ndarray): Convex hull vertices from compute_convex_hull()
        
        Returns:
            List[Tuple[int, int]]: List of (x, y) fingertip coordinates
        
        Notes:
            - Returns up to max_finger_count positions
            - For closed fist, may return palm/wrist points
            - Limit to most extreme points if needed
        """
        if hull is None or len(hull) == 0:
            return []
        
        fingertips = []
        for point in hull:
            x, y = point[0]
            fingertips.append((int(x), int(y)))
        
        # Limit to max finger count (remove some less extreme points)
        if len(fingertips) > self.max_finger_count:
            # Sort by distance from centroid to keep periphery points
            centroid = np.mean(hull.reshape(-1, 2), axis=0)
            distances = [np.linalg.norm(pt - centroid) for pt in hull.reshape(-1, 2)]
            indices = np.argsort(distances)[-self.max_finger_count:]
            fingertips = [fingertips[i] for i in sorted(indices)]
        
        return fingertips
    
    def extract_features(self, contour: np.ndarray) -> dict:
        """
        Complete feature extraction pipeline.
        
        Sequence:
            1. Approximate contour
            2. Compute convex hull
            3. Compute convexity defects
            4. Filter defects by angle
            5. Count fingers
            6. Extract fingertip positions
        
        Args:
            contour (np.ndarray): Hand contour from Module 2
        
        Returns:
            dict: Feature dictionary with:
                - 'finger_count': Estimated count (0-5)
                - 'defects': List of valid defect dicts
                - 'hull': Convex hull vertices
                - 'fingertips': List of (x, y) coordinates
                - 'hull_area': Area of convex hull
                - 'contour_area': Area of hand contour
                - 'solidity': Ratio of contour to hull area
        
        Time Complexity: O(n log n) dominated by hull computation
        Expected Duration: 3-5 ms
        """
        # Step 1: Approximate contour
        approx_contour = self.approximate_contour(contour)
        
        # Step 2: Compute convex hull
        hull = self.compute_convex_hull(approx_contour)
        
        # Step 3: Compute convexity defects
        defects = self.compute_convexity_defects(approx_contour, hull)
        
        # Step 4: Filter defects by angle
        valid_defects = []
        if defects is not None:
            valid_defects = self.filter_defects_by_angle(approx_contour, defects)
        
        # Step 5: Count fingers
        finger_count = self.estimate_finger_count(valid_defects)
        
        # Step 6: Get fingertip positions
        fingertips = self.get_fingertip_positions(hull)
        
        # Calculate solidity metrics
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        
        return {
            'finger_count': finger_count,
            'defects': valid_defects,
            'hull': hull,
            'fingertips': fingertips,
            'hull_area': hull_area,
            'contour_area': contour_area,
            'solidity': solidity,
            'num_defects': len(valid_defects)
        }


if __name__ == "__main__":
    """
    Test Module 3 independently.
    
    Requires Module 1 and Module 2 to be available.
    """
    print("=" * 60)
    print("MODULE 3: FEATURE EXTRACTION TEST")
    print("=" * 60)
    
    try:
        from video_capture_preprocessing import VideoCapturePreprocessor
        from hand_detection_segmentation import HandDetectionSegmentor
        
        # Initialize modules
        preprocessor = VideoCapturePreprocessor()
        detector = HandDetectionSegmentor()
        extractor = FeatureExtractor()
        
        print("\nStarting feature extraction test (press 'q' to quit)...")
        print("Show different hand gestures (open palm, fist, peace sign, etc.)\n")
        
        frame_count = 0
        gesture_history = {}
        
        while True:
            ret, raw_frame = preprocessor.read_frame()
            if not ret:
                break
            
            # Module 1: Preprocess
            bgr_frame, hsv_frame = preprocessor.preprocess_frame(raw_frame)
            
            # Module 2: Detect hand
            hand_contour, mask, detection_props = detector.detect_hand(hsv_frame)
            
            if not detection_props['found']:
                cv2.imshow("Feature Extraction", bgr_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Module 3: Extract features
            features = extractor.extract_features(hand_contour)
            finger_count = features['finger_count']
            
            # Track gesture frequency
            gesture_history[finger_count] = gesture_history.get(finger_count, 0) + 1
            frame_count += 1
            
            if frame_count % 10 == 0:  # Print every 10 frames
                print(f"Frame {frame_count}: Fingers detected = {finger_count}, " +
                      f"Defects found = {features['num_defects']}, " +
                      f"Fingertips = {len(features['fingertips'])}")
            
            # Visualize
            output = bgr_frame.copy()
            
            # Draw hand contour
            if hand_contour is not None:
                cv2.drawContours(output, [hand_contour], 0, (0, 255, 0), 2)
            
            # Draw convex hull
            if features['hull'] is not None:
                cv2.drawContours(output, [features['hull']], 0, (255, 0, 0), 1)
            
            # Draw fingertips
            for fp in features['fingertips']:
                cv2.circle(output, fp, 5, (0, 0, 255), -1)
            
            # Draw defects
            if features['hull'] is not None and hand_contour is not None:
                for defect in features['defects']:
                    pt = defect['farthest']
                    cv2.circle(output, pt, 4, (0, 255, 255), -1)
            
            # Add text information
            cv2.putText(output, f"Fingers: {finger_count}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(output, f"Defects: {features['num_defects']}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output, f"Solidity: {features['solidity']:.2f}", (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("Feature Extraction", output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\nResults Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Gesture distribution:")
        for fingers in sorted(gesture_history.keys()):
            count = gesture_history[fingers]
            percentage = 100 * count / frame_count
            print(f"    {fingers} fingers: {count} frames ({percentage:.1f}%)")
        
        preprocessor.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
