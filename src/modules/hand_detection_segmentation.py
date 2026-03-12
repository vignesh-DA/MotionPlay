"""
MODULE 2: HAND DETECTION AND SEGMENTATION
==========================================

Purpose:
    Isolate the hand region from the background using HSV color-based segmentation,
    producing a binary mask identifying hand pixels and extracting the hand contour.

Responsibilities:
    - Apply HSV color thresholding for skin detection
    - Perform morphological operations (opening, closing)
    - Detect hand contour from binary mask
    - Select largest contour (actual hand)
    - Extract hand bounding box and centroid

Input:  HSV frame from Module 1, preprocessed BGR frame
Output: Binary mask, hand contour, hand region properties

Author:     Capstone Project
Date:       2025
Version:    1.0
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
from datetime import datetime
from pathlib import Path


class HandDetectionSegmentor:
    """
    Detects hand region from HSV frame using color-based segmentation.
    
    Attributes:
        hsv_lower (np.ndarray): Lower HSV threshold [H, S, V]
        hsv_upper (np.ndarray): Upper HSV threshold [H, S, V]
        min_contour_area (int): Minimum area for valid hand (pixels)
        min_contour_solidity (float): Minimum solidity filter (0-1)
    """
    
    def __init__(self, 
                 hsv_lower: Tuple[int, int, int] = (0, 30, 80),
                 hsv_upper: Tuple[int, int, int] = (20, 150, 255),
                 min_contour_area: int = 20,
                 min_contour_solidity: float = 0.0):
        """
        Initialize hand detection parameters.
        
        Args:
            hsv_lower (Tuple): Lower HSV bounds [H, S, V]
                              Default: (0, 30, 80) - Red skin tones
            hsv_upper (Tuple): Upper HSV bounds [H, S, V]
                              Default: (20, 150, 255)
            min_contour_area (int): Minimum area for hand (default 200px - ultra-low for maximum sensitivity)
            min_contour_solidity (float): Minimum solidity ratio (default 0.2 - extremely permissive for fragmented hands)
        
        Note:
            HSV ranges are empirically determined for skin color detection.
            H: Hue (0-180 in OpenCV)
            S: Saturation (0-255)
            V: Value/Brightness (0-255)
        """
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_contour_area = min_contour_area
        self.min_contour_solidity = min_contour_solidity
        
        # Create structuring elements for morphological operations
        # Elliptical kernel works better for hand shapes than rectangular
        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)
        )
        
        # Initialize logging for hand detection diagnostics
        self.frame_count = 0
        self.detection_count = 0
        self.last_hand_detected = False
        self.detection_history = []
        
        # Setup logging file
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        log_filename = self.log_dir / f"hand_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logger
        self.logger = logging.getLogger('HandDetection')
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler (for errors only) - with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*70)
        self.logger.info("HAND DETECTION AND SEGMENTATION - INITIALIZATION")
        self.logger.info("="*70)
        self.logger.info(f"[OK] Hand detection segmentor initialized")
        self.logger.info(f"  HSV Range: H[{self.hsv_lower[0]}-{self.hsv_upper[0]}], " +
                        f"S[{self.hsv_lower[1]}-{self.hsv_upper[1]}], " +
                        f"V[{self.hsv_lower[2]}-{self.hsv_upper[2]}]")
        self.logger.info(f"  Min contour area: {min_contour_area} pixels")
        self.logger.info(f"  Min contour solidity: {min_contour_solidity}")
        self.logger.info(f"  Log file: {log_filename}")
        
        print("[OK] Hand detection segmentor initialized")
        print(f"  HSV Range: H{self.hsv_lower[0]}-{self.hsv_upper[0]}, " +
              f"S{self.hsv_lower[1]}-{self.hsv_upper[1]}, " +
              f"V{self.hsv_lower[2]}-{self.hsv_upper[2]}")
        print(f"  [LOG] Detection log: {log_filename}")
    
    def apply_hsv_threshold(self, hsv_frame: np.ndarray) -> np.ndarray:
        """
        Apply HSV color thresholding to create binary mask.
        
        Algorithm:
            1. Apply cv2.inRange() to extract pixels within HSV bounds
            2. Result is binary mask: white (255) = skin, black (0) = background
        
        Mathematical Basis:
            mask[x,y] = 255 if (hsv_lower <= hsv[x,y] <= hsv_upper)
                      = 0   otherwise
        
        Args:
            hsv_frame (np.ndarray): HSV frame from Module 1
                                   shape (480, 640, 3), dtype uint8
        
        Returns:
            np.ndarray: Binary mask, shape (480, 640), dtype uint8
                       White (255) = skin pixels, Black (0) = background
        
        Time Complexity: O(n) where n = number of pixels (~307k)
        Expected Duration: 1-2 ms
        """
        # cv2.inRange: Creates binary mask in one pass
        # All three HSV channels must be within bounds simultaneously
        mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
        
        return mask
    
    def morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean segmentation mask.
        
        Two-step process:
        
        Step 1: Morphological Opening
            Opening = Erosion followed by Dilation
            Effect: Removes small foreground objects (noise speckles)
            Preserves: Larger structures (actual hand)
        
        Step 2: Morphological Closing
            Closing = Dilation followed by Erosion
            Effect: Fills small background holes (within hand region)
            Result: Solid, continuous hand region
        
        Args:
            mask (np.ndarray): Binary mask from HSV thresholding
                              shape (480, 640), dtype uint8
        
        Returns:
            np.ndarray: Cleaned binary mask, shape (480, 640), dtype uint8
        
        Time Complexity: O(n × k) where n = pixels, k = kernel size
        Expected Duration: 2-3 ms
        """
        # Step 1: Opening (remove external noise)
        # cv2.MORPH_OPEN = erosion followed by dilation
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        
        # Step 2: Closing (fill internal holes)
        # cv2.MORPH_CLOSE = dilation followed by erosion
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel_close)
        
        return closing
    
    def find_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand contour from binary mask.
        
        Algorithm:
            1. Find all contours via cv2.findContours()
            2. Filter by minimum area
            3. Select largest contour (likely to be hand)
            4. Validate contour shape (solidity check)
        
        Args:
            mask (np.ndarray): Binary mask from morphological operations
                              shape (480, 640), dtype uint8
        
        Returns:
            Optional[np.ndarray]: Hand contour coordinates
                                 shape (N, 1, 2), dtype int32 (OpenCV format)
                                 None if no valid contour found
        
        Time Complexity: O(n log n) for contour detection
        Expected Duration: 1-2 ms
        """
        # Find contours in mask
        # cv2.RETR_EXTERNAL: Only retrieve external contours (no holes)
        # cv2.CHAIN_APPROX_SIMPLE: Compress contours (reduce points)
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Filter contours by area and find largest
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
        
        # Sort by area, get largest
        hand_contour, max_area = max(valid_contours, key=lambda x: x[1])
        
        # Validate contour solidity
        # Solidity = Area / ConvexArea
        # Measures how "solid" shape is (vs. fragmented)
        hull = cv2.convexHull(hand_contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < self.min_contour_solidity:
                return None  # Shape is too fragmented
        
        return hand_contour
    
    def get_hand_bounding_box(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get axis-aligned bounding box around hand contour.
        
        Args:
            contour (np.ndarray): Hand contour from find_hand_contour()
        
        Returns:
            Tuple[int, int, int, int]: (x, y, width, height)
                - x, y: Top-left corner coordinates
                - width, height: Bounding box dimensions
        """
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h
    
    def get_hand_centroid(self, contour: np.ndarray) -> Tuple[float, float]:
        """
        Calculate hand centroid (center of mass) from contour.
        
        Uses image moments: Mathematical property where centroid is the
        first-order moment divided by zero-order moment (area).
        
        Args:
            contour (np.ndarray): Hand contour
        
        Returns:
            Tuple[float, float]: (cx, cy) centroid coordinates
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0, 0
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        return cx, cy
    
    def detect_hand(self, hsv_frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
        """
        Complete hand detection and segmentation pipeline.
        
        Runs all steps sequentially:
            1. HSV thresholding
            2. Morphological operations
            3. Contour detection
            4. Hand property extraction
        
        Args:
            hsv_frame (np.ndarray): HSV frame from Module 1
        
        Returns:
            Tuple with three elements:
                - hand_contour: Detected contour (None if not found)
                - mask: Binary segmentation mask
                - properties: Dictionary with hand properties:
                    - 'found': bool (hand detected)
                    - 'bbox': (x, y, w, h) bounding box
                    - 'centroid': (cx, cy) center of mass
                    - 'area': contour area in pixels
                    - 'x', 'y', 'w', 'h': Components of bounding box
        
        Time Complexity: O(n log n)
        Expected Duration: 5-8 ms
        """
        self.frame_count += 1
        
        # Step 1: HSV thresholding
        mask = self.apply_hsv_threshold(hsv_frame)
        mask_pixels = np.count_nonzero(mask)
        mask_percentage = (mask_pixels / mask.size) * 100
        
        # Step 2: Morphological operations
        mask = self.morphological_operations(mask)
        cleaned_pixels = np.count_nonzero(mask)
        cleaned_percentage = (cleaned_pixels / mask.size) * 100
        
        # Step 3: Find hand contour
        hand_contour = self.find_hand_contour(mask)
        
        # Step 4: Extract properties
        properties = {'found': False}
        
        if hand_contour is not None:
            properties['found'] = True
            properties['bbox'] = self.get_hand_bounding_box(hand_contour)
            properties['x'], properties['y'], properties['w'], properties['h'] = properties['bbox']
            properties['centroid'] = self.get_hand_centroid(hand_contour)
            properties['area'] = cv2.contourArea(hand_contour)
            
            # Log successful detection
            self.detection_count += 1
            status = "[OK] DETECTED"
            self.logger.info(f"Frame {self.frame_count}: {status}")
            self.logger.info(f"  - Hand contour area: {properties['area']:.0f} pixels")
            self.logger.info(f"  - Bounding box: x={properties['x']}, y={properties['y']}, " +
                           f"w={properties['w']}, h={properties['h']}")
            self.logger.info(f"  - Centroid: ({properties['centroid'][0]:.1f}, " +
                           f"{properties['centroid'][1]:.1f})")
            self.logger.info(f"  - Detection rate: {100*self.detection_count/self.frame_count:.1f}%")
            
            self.last_hand_detected = True
        else:
            # Log detection failure with diagnostic info
            status = "[FAIL] NOT DETECTED"
            self.logger.warning(f"Frame {self.frame_count}: {status}")
            self.logger.warning(f"  - Reason: No valid hand contour found")
            self.logger.warning(f"  - HSV mask pixels: {mask_pixels} ({mask_percentage:.2f}%)")
            self.logger.warning(f"  - After cleaning: {cleaned_pixels} ({cleaned_percentage:.2f}%)")
            self.logger.warning(f"  - Detection rate: {100*self.detection_count/self.frame_count:.1f}%")
            
            self.last_hand_detected = False
        
        # Store in history for analysis
        self.detection_history.append({
            'frame': self.frame_count,
            'detected': properties['found'],
            'mask_pixels': mask_pixels,
            'cleaned_pixels': cleaned_pixels,
            'timestamp': datetime.now()
        })
        
        return hand_contour, mask, properties
    
    def update_hsv_range(self, 
                        hsv_lower: Tuple[int, int, int],
                        hsv_upper: Tuple[int, int, int]) -> None:
        """
        Update HSV threshold range (useful for calibration).
        
        Args:
            hsv_lower (Tuple): New lower HSV bounds
            hsv_upper (Tuple): New upper HSV bounds
        """
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.logger.info(f"[OK] HSV range updated: H[{hsv_lower[0]}-{hsv_upper[0]}], " +
                        f"S[{hsv_lower[1]}-{hsv_upper[1]}], " +
                        f"V[{hsv_lower[2]}-{hsv_upper[2]}]")
        print(f"[OK] HSV range updated: H{hsv_lower[0]}-{hsv_upper[0]}")
    
    def get_detection_statistics(self) -> dict:
        """
        Get hand detection statistics.
        
        Returns:
            dict: Statistics including:
                - 'total_frames': Total frames processed
                - 'detections': Total hands detected
                - 'detection_rate': Percentage of frames with hand
                - 'last_detected': Whether hand was detected in last frame
        """
        rate = 100 * self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'detections': self.detection_count,
            'detection_rate': rate,
            'last_detected': self.last_hand_detected
        }
    
    def log_diagnostics(self, bgr_frame: np.ndarray, hsv_frame: np.ndarray) -> None:
        """
        Log detailed frame diagnostics for troubleshooting.
        
        Args:
            bgr_frame (np.ndarray): Original BGR frame
            hsv_frame (np.ndarray): Converted HSV frame
        """
        # Calculate frame statistics
        h, w = bgr_frame.shape[:2]
        
        # BGR statistics
        bgr_blue_mean = bgr_frame[:,:,0].mean()
        bgr_green_mean = bgr_frame[:,:,1].mean()
        bgr_red_mean = bgr_frame[:,:,2].mean()
        
        # HSV statistics
        hsv_hue_mean = hsv_frame[:,:,0].mean()
        hsv_sat_mean = hsv_frame[:,:,1].mean()
        hsv_val_mean = hsv_frame[:,:,2].mean()
        
        self.logger.debug(f"Frame Size: {w}x{h}")
        self.logger.debug(f"  BGR means - B:{bgr_blue_mean:.1f}, G:{bgr_green_mean:.1f}, R:{bgr_red_mean:.1f}")
        self.logger.debug(f"  HSV means - H:{hsv_hue_mean:.1f}, S:{hsv_sat_mean:.1f}, V:{hsv_val_mean:.1f}")


if __name__ == "__main__":
    """
    Test Module 2 independently.
    
    Requires a real-time camera feed.
    """
    print("=" * 60)
    print("MODULE 2: HAND DETECTION & SEGMENTATION TEST")
    print("=" * 60)
    
    try:
        from video_capture_preprocessing import VideoCapturePreprocessor
        
        # Initialize both modules
        preprocessor = VideoCapturePreprocessor()
        detector = HandDetectionSegmentor()
        
        print("\nStarting live detection (press 'q' to quit)...")
        print("Point your hand at the camera.\n")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, raw_frame = preprocessor.read_frame()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Apply preprocessing
            bgr_frame, hsv_frame = preprocessor.preprocess_frame(raw_frame)
            
            # Detect hand
            hand_contour, mask, properties = detector.detect_hand(hsv_frame)
            
            frame_count += 1
            if properties['found']:
                detection_count += 1
                bbox = properties['bbox']
                area = properties['area']
                print(f"Frame {frame_count}: Hand detected! " +
                      f"Area={area:.0f}px, Bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
            
            # Visualize
            output = bgr_frame.copy()
            
            if hand_contour is not None:
                # Draw contour
                cv2.drawContours(output, [hand_contour], 0, (0, 255, 0), 2)
                
                # Draw bounding box
                x, y, w, h = properties['bbox']
                cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw centroid
                cx, cy = properties['centroid']
                cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            
            # Stack frames for display
            display = np.hstack([output, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            
            cv2.imshow("Hand Detection (Left: RGB, Right: Mask)", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        print(f"\nResults:")
        print(f"  Total frames: {frame_count}")
        print(f"  Hands detected: {detection_count}")
        print(f"  Detection rate: {100*detection_count/frame_count:.1f}%")
        
        preprocessor.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
