"""
MODULE 1: VIDEO CAPTURE AND PREPROCESSING
==========================================

Purpose:
    Acquire raw video frames from the webcam and prepare them for hand detection
    by reducing noise and improving image quality.

Responsibilities:
    - Initialize and manage video capture from webcam
    - Resize frames to standard resolution (640×480)
    - Apply Gaussian blur for noise reduction
    - Apply histogram equalization (CLAHE) for lighting adaptation
    - Prepare frames for hand detection in next module

Input:  Webcam video stream (variable resolution, ~30 FPS)
Output: Preprocessed frames (BGR and HSV), metadata

Author:     Capstone Project
Date:       2025
Version:    1.0
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class VideoCapturePreprocessor:
    """
    Handles video capture initialization, frame preprocessing, and quality enhancement.
    
    Attributes:
        device_index (int): Webcam device index (usually 0 for primary device)
        frame_width (int): Target frame width in pixels (640)
        frame_height (int): Target frame height in pixels (480)
        target_fps (int): Target frames per second
        
        cap (cv2.VideoCapture): OpenCV VideoCapture object
        clahe (cv2.CLAHE): CLAHE object for histogram equalization
        frame_count (int): Total frames processed counter
        is_initialized (bool): Initialization status flag
    """
    
    def __init__(self, device_index: int = 0, frame_width: int = 640, 
                 frame_height: int = 480, target_fps: int = 30):
        """
        Initialize video capture and preprocessing parameters.
        
        Args:
            device_index (int): Webcam index (0 = default/primary)
            frame_width (int): Target frame width (default 640)
            frame_height (int): Target frame height (default 480)
            target_fps (int): Target frames per second (default 30)
        
        Raises:
            RuntimeError: If webcam cannot be initialized
        """
        self.device_index = device_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        
        # Initialize frame counter
        self.frame_count = 0
        self.is_initialized = False
        
        # Initialize VideoCapture
        self.cap = cv2.VideoCapture(device_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam device {device_index}")
        
        # Set camera properties
        self._configure_camera()
        
        # Initialize CLAHE for histogram equalization
        # clipLimit: Contrast limitation threshold (2.0 = moderate)
        # tileGridSize: Size of tiles for local processing (8x8)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        self.is_initialized = True
        print(f"✓ Video capture initialized: {frame_width}×{frame_height} @ {target_fps} FPS")
    
    def _configure_camera(self) -> None:
        """
        Configure camera properties for optimal performance.
        
        Sets:
            - Frame width and height
            - FPS setting
            - Exposure and brightness (if supported)
        """
        # Set frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Attempt to improve image quality (may not work on all cameras)
        try:
            # Disable auto-focus if possible
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except:
            pass
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the webcam.
        
        Returns:
            Tuple[bool, np.ndarray]: (success, frame)
                - success: True if frame captured successfully
                - frame: BGR frame array, shape (height, width, 3), dtype uint8
                         None if capture failed
        
        Note:
            Raw frame may have variable resolution depending on camera
        """
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
        
        return ret, frame
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply preprocessing pipeline to raw frame.
        
        Processing steps:
            1. Resize to standard resolution (640×480)
            2. Apply Gaussian Blur (5×5 kernel, σ=1.0)
               Purpose: Reduce noise and smooth lighting variations
            3. Apply CLAHE (histogram equalization)
               Purpose: Enhance local contrast, adapt to lighting
            4. Convert BGR → HSV for next module
        
        Args:
            frame (np.ndarray): Raw BGR frame from webcam
                               shape (H, W, 3), dtype uint8
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - preprocessed_bgr: Preprocessed BGR frame (640×480)
                - hsv_frame: HSV-converted frame for hand detection
                
        Time Complexity: O(n) where n = number of pixels (~307k for 640×480)
        Expected Duration: 5-8 ms per frame
        """
        # Step 1: Resize frame
        # Bilinear interpolation for shrinking, cubic for enlarging
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height),
                             interpolation=cv2.INTER_LINEAR)
        
        # Step 2: Apply Gaussian Blur
        # Kernel size: 5×5 (must be odd number)
        # Sigma: 1.0 (standard deviation of Gaussian)
        # Purpose: Reduce high-frequency noise, smooth lighting gradients
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        
        # Step 3: Apply histogram equalization (CLAHE)
        # Convert BGR → LAB color space for luminance equalization
        # Process only L channel (luminance) to avoid color distortion
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        l_equalized = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
        
        # Convert back to BGR
        preprocessed_bgr = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        
        # Step 4: Convert BGR → HSV for hand detection
        # HSV separates color (Hue) from brightness (Value)
        # Better suited for color-based segmentation than BGR/RGB
        hsv_frame = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2HSV)
        
        return preprocessed_bgr, hsv_frame
    
    def initialize_with_frame_capture(self) -> bool:
        """
        Test camera initialization by capturing a frame.
        
        Useful to ensure camera is working before main processing loop.
        
        Returns:
            bool: True if frame captured successfully, False otherwise
        """
        ret, frame = self.read_frame()
        
        if ret:
            print(f"✓ Camera frame captured successfully: {frame.shape}")
            return True
        else:
            print("✗ Failed to capture frame from camera")
            return False
    
    def release(self) -> None:
        """
        Release video capture resources.
        
        Should be called when exiting the application to clean up resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.is_initialized = False
            print("✓ Video capture released")
    
    def get_camera_properties(self) -> dict:
        """
        Retrieve current camera properties.
        
        Returns:
            dict: Dictionary with camera properties:
                - width: Current frame width
                - height: Current frame height
                - fps: Current FPS setting
                - frame_count: Total frames captured
                - is_open: Whether camera is available
        """
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': self.frame_count,
            'is_open': self.cap.isOpened()
        }
    
    def __del__(self):
        """Destructor: Ensure resources are released."""
        self.release()


if __name__ == "__main__":
    """
    Test Module 1 independently.
    
    Tests:
        1. Camera initialization
        2. Frame capture
        3. Preprocessing pipeline
        4. Frame output visualization
    """
    print("=" * 60)
    print("MODULE 1: VIDEO CAPTURE & PREPROCESSING TEST")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = VideoCapturePreprocessor(
            device_index=0,
            frame_width=640,
            frame_height=480,
            target_fps=30
        )
        
        # Print camera properties
        print("\nCamera Properties:")
        props = preprocessor.get_camera_properties()
        for key, value in props.items():
            print(f"  {key}: {value}")
        
        # Capture and process test frame
        print("\nCapturing test frame...")
        ret, raw_frame = preprocessor.read_frame()
        
        if ret:
            print(f"✓ Frame captured: {raw_frame.shape}")
            
            # Apply preprocessing
            preprocessed_bgr, hsv_frame = preprocessor.preprocess_frame(raw_frame)
            
            print(f"✓ Preprocessing complete:")
            print(f"  - BGR output shape: {preprocessed_bgr.shape}")
            print(f"  - HSV output shape: {hsv_frame.shape}")
            
            # Show statistics
            print(f"\nFrame Statistics:")
            print(f"  BGR mean intensity: {np.mean(preprocessed_bgr):.2f}")
            print(f"  BGR std deviation: {np.std(preprocessed_bgr):.2f}")
            print(f"  HSV H range: {hsv_frame[:,:,0].min()}-{hsv_frame[:,:,0].max()}")
            
            # Test multiple frames for timing
            print(f"\nProcessing 30 frames for timing benchmark...")
            import time
            start_time = time.time()
            
            for _ in range(30):
                ret, raw = preprocessor.read_frame()
                if ret:
                    _ = preprocessor.preprocess_frame(raw)
            
            elapsed = time.time() - start_time
            avg_time = (elapsed / 30) * 1000  # Convert to ms
            fps = 30 / elapsed
            
            print(f"✓ Benchmark Results:")
            print(f"  - Average time per frame: {avg_time:.2f} ms")
            print(f"  - Achieved FPS: {fps:.1f}")
            
        else:
            print("✗ Failed to capture frame")
        
        preprocessor.release()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
