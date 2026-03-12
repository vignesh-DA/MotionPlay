"""
HAND DETECTION DIAGNOSTIC TOOL
===============================

Purpose:
    Analyze video frames to troubleshoot hand detection failures.
    Helps identify HSV range issues, lighting problems, and configuration needs.

Usage:
    python diagnose_hand_detection.py

This tool will:
  1. Open your webcam
  2. Show real-time HSV statistics of the frame
  3. Display what colors are currently masked
  4. Suggest HSV range adjustments
  5. Save a diagnostic report

Author:     Capstone Project
Date:       2025
"""

import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime


class HandDetectionDiagnostic:
    """Diagnostic tool for hand detection troubleshooting."""
    
    def __init__(self):
        """Initialize diagnostic tool."""
        self.cap = None
        self.log_file = None
        self.frame_count = 0
        self.hsv_readings = []
        
    def open_camera(self, camera_index=0):
        """Open camera device."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("✗ Failed to open camera device")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Camera opened successfully")
        return True
    
    def setup_logging(self):
        """Setup diagnostic log file."""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.log_file = log_dir / f"hand_detection_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(self.log_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HAND DETECTION DIAGNOSTIC REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
        
        print(f"📄 Log file: {self.log_file}")
        return True
    
    def capture_hand_area(self):
        """
        Capture samples of hand area for analysis.
        Guides user to position hand in frame.
        """
        print("\n" + "="*70)
        print("HAND DETECTION DIAGNOSTIC - CAPTURE PHASE")
        print("="*70)
        print("\nInstructions:")
        print("  1. Press SPACE to capture a frame when your hand is visible")
        print("  2. Try different hand positions, lighting, and distances")
        print("  3. Capture at least 3-5 different hand positions")
        print("  4. Press 'q' when done capturing")
        print("\nWhat the tool shows:")
        print("  LEFT:   Original frame")
        print("  MIDDLE: HSV Color range mask for skin (H:0-20)")
        print("  RIGHT:  After morphological cleaning")
        print("\nColor indicators:")
        print("  GREEN = pixels likely to be skin color (within H:0-20)")
        print("  BLACK = pixels NOT likely to be skin (outside range)")
        
        captured_frames = []
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Create skin color mask (default HSV range)
                # Hue: 0-20 (red skin tones)
                # Saturation: 30-150
                # Value: 80-255
                lower_skin = np.array([0, 30, 80], dtype=np.uint8)
                upper_skin = np.array([20, 150, 255], dtype=np.uint8)
                
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                
                # Apply morphological operations
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                
                opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                cleaned = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
                
                # Convert masks to BGR for display
                mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cleaned_display = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                
                # Create display
                h, w = frame.shape[:2]
                display = np.hstack([
                    frame,
                    mask_display,
                    cleaned_display
                ])
                
                # Add statistics overlay
                mask_pixels = np.count_nonzero(mask)
                cleaned_pixels = np.count_nonzero(cleaned)
                total_pixels = mask.size
                
                text_lines = [
                    f"Frame: {self.frame_count}",
                    f"HSV Mask: {mask_pixels} pixels ({100*mask_pixels/total_pixels:.1f}%)",
                    f"Cleaned: {cleaned_pixels} pixels ({100*cleaned_pixels/total_pixels:.1f}%)",
                    f"SPACE=capture, q=done"
                ]
                
                for i, text in enumerate(text_lines):
                    cv2.putText(display, text, (10, 30 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Get HSV statistics in center region (likely hand)
                h_mid, w_mid = h//2, w//2
                roi = hsv[h_mid-50:h_mid+50, w_mid-50:w_mid+50]
                
                if roi.size > 0:
                    h_mean = roi[:,:,0].mean()
                    s_mean = roi[:,:,1].mean()
                    v_mean = roi[:,:,2].mean()
                    
                    stats_text = f"Center HSV: H={h_mean:.0f}, S={s_mean:.0f}, V={v_mean:.0f}"
                    cv2.putText(display, stats_text, (10, display.shape[0]-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    self.hsv_readings.append({
                        'h': h_mean, 's': s_mean, 'v': v_mean,
                        'timestamp': time.time()
                    })
                
                # Display
                cv2.imshow("Hand Detection Diagnosis", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Save frame
                    captured_frames.append({
                        'frame': frame,
                        'hsv': hsv,
                        'mask': mask,
                        'cleaned': cleaned,
                        'timestamp': datetime.now()
                    })
                    print(f"✓ Captured frame {len(captured_frames)}")
                    
                elif key == ord('q'):
                    break
        
        except Exception as e:
            print(f"✗ Error during capture: {e}")
        
        cv2.destroyAllWindows()
        return captured_frames
    
    def analyze_captures(self, captured_frames):
        """Analyze captured frames and generate recommendations."""
        if not captured_frames:
            print("✗ No frames captured for analysis")
            return
        
        print("\n" + "="*70)
        print("DIAGNOSTIC ANALYSIS")
        print("="*70)
        
        all_h = []
        all_s = []
        all_v = []
        
        for i, capture in enumerate(captured_frames):
            hsv = capture['hsv']
            mask = capture['mask']
            
            # Get HSV statistics
            h = hsv[:,:,0]
            s = hsv[:,:,1]
            v = hsv[:,:,2]
            
            # Focus on skin regions found
            mask_pixels = np.count_nonzero(mask)
            if mask_pixels > 0:
                skin_h = h[mask > 0]
                skin_s = s[mask > 0]
                skin_v = v[mask > 0]
                
                all_h.extend(skin_h.flatten())
                all_s.extend(skin_s.flatten())
                all_v.extend(skin_v.flatten())
                
                print(f"\nFrame {i+1} (pixels detected: {mask_pixels}):")
                print(f"  Hue range:       {skin_h.min()}-{skin_h.max()}")
                print(f"  Saturation range: {skin_s.min()}-{skin_s.max()}")
                print(f"  Value range:      {skin_v.min()}-{skin_v.max()}")
            else:
                print(f"\nFrame {i+1}: ✗ NO SKIN PIXELS DETECTED")
                
                # Analyze entire frame
                print(f"  Full frame HSV stats:")
                print(f"    Hue: min={h.min()}, max={h.max()}, mean={h.mean():.0f}")
                print(f"    Sat: min={s.min()}, max={s.max()}, mean={s.mean():.0f}")
                print(f"    Val: min={v.min()}, max={v.max()}, mean={v.mean():.0f}")
        
        # Overall analysis
        print("\n" + "-"*70)
        print("OVERALL ANALYSIS")
        print("-"*70)
        
        if all_h:
            print(f"\nAll detected skin pixels:")
            print(f"  Hue:       {int(np.min(all_h))}-{int(np.max(all_h))} (current: 0-20)")
            print(f"  Saturation: {int(np.min(all_s))}-{int(np.max(all_s))} (current: 30-150)")
            print(f"  Value:      {int(np.min(all_v))}-{int(np.max(all_v))} (current: 80-255)")
            
            # Recommendations
            print(f"\n📋 RECOMMENDATIONS:")
            
            # Check Hue
            h_min, h_max = int(np.min(all_h)), int(np.max(all_h))
            if h_max > 20:
                print(f"  ⚠️  Your hand needs wider Hue range: 0-{h_max} (wider than default 0-20)")
                print(f"      This suggests you have more orange/yellow tones in hand")
            
            # Check Saturation
            s_min, s_max = int(np.min(all_s)), int(np.max(all_s))
            if s_min < 30:
                print(f"  ⚠️  Lower saturation detected: {s_min}-{s_max} (wider than default 30-150)")
                print(f"      Consider allowing lower saturation: {max(0, s_min-10)}-150")
            
            # Check Value
            v_min, v_max = int(np.min(all_v)), int(np.max(all_v))
            if v_min > 80:
                print(f"  ℹ️  Value range is well within threshold (good lighting)")
            elif v_min < 50:
                print(f"  ⚠️  Low brightness detected: {v_min}-{v_max}")
                print(f"      Try better lighting or modify Value lower bound to {max(0, v_min-20)}")
            
            # Save recommendations
            recommended_h_lower = max(0, h_min - 5)
            recommended_h_upper = min(180, h_max + 5)
            recommended_s_lower = max(0, s_min - 10)
            recommended_s_upper = min(255, s_max + 10)
            recommended_v_lower = max(0, v_min - 10)
            recommended_v_upper = min(255, v_max + 10)
            
            print(f"\n✅ SUGGESTED HSV RANGE:")
            print(f"  Lower: ({recommended_h_lower}, {recommended_s_lower}, {recommended_v_lower})")
            print(f"  Upper: ({recommended_h_upper}, {recommended_s_upper}, {recommended_v_upper})")
            
        else:
            print("\n✗ ERROR: No skin pixels detected in any captured frame!")
            print("\nPossible causes:")
            print("  1. Wrong camera index (try different camera device)")
            print("  2. Hand not positioned correctly in frame")
            print("  3. Extreme lighting conditions (too bright/dark)")
            print("  4. Camera quality issues")
            print("\nNext steps:")
            print("  - Ensure hand is clearly visible in frame")
            print("  - Check lighting (not too bright, not too dark)")
            print("  - Position hand 20-60 cm from camera")
            print("  - Try different camera if available")
        
        # Save to log file
        self._save_analysis_to_log(captured_frames, all_h, all_s, all_v)
    
    def _save_analysis_to_log(self, frames, h_vals, s_vals, v_vals):
        """Save analysis results to log file."""
        if not self.log_file:
            return
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nFrames captured: {len(frames)}\n")
            f.write(f"Total frames processed: {self.frame_count}\n\n")
            
            if h_vals:
                f.write("HSV Statistics from detected skin pixels:\n")
                f.write(f"  Hue:       {int(np.min(h_vals))}-{int(np.max(h_vals))}\n")
                f.write(f"  Saturation: {int(np.min(s_vals))}-{int(np.max(s_vals))}\n")
                f.write(f"  Value:      {int(np.min(v_vals))}-{int(np.max(v_vals))}\n\n")
                
                f.write("Recommended HSV Range:\n")
                f.write(f"  Lower: ({max(0, int(np.min(h_vals))-5)}, ")
                f.write(f"{max(0, int(np.min(s_vals))-10)}, ")
                f.write(f"{max(0, int(np.min(v_vals))-10)})\n")
                f.write(f"  Upper: ({min(180, int(np.max(h_vals))+5)}, ")
                f.write(f"{min(255, int(np.max(s_vals))+10)}, ")
                f.write(f"{min(255, int(np.max(v_vals))+10)})\n")
            else:
                f.write("ERROR: No skin pixels detected in any frame\n")
            
            f.write(f"\nReport saved: {datetime.now()}\n")
    
    def run(self):
        """Run complete diagnostic."""
        print("\n" + "="*70)
        print("HAND DETECTION TROUBLESHOOTING TOOL")
        print("="*70 + "\n")
        
        if not self.open_camera():
            return
        
        self.setup_logging()
        
        # Capture frames
        captured_frames = self.capture_hand_area()
        
        # Analyze
        self.analyze_captures(captured_frames)
        
        print(f"\n✓ Diagnostic complete!")
        print(f"📄 Full report saved to: {self.log_file}")
        
        if self.cap:
            self.cap.release()


def main():
    """Main entry point."""
    try:
        diagnostic = HandDetectionDiagnostic()
        diagnostic.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
