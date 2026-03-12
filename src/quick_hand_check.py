#!/usr/bin/env python3
"""
QUICK HAND DETECTION CHECK
===========================

Fast diagnostic to see if hand detection is working.
Run this before running the full application.

Usage:
    python quick_hand_check.py

This script will:
  1. Check camera access
  2. Test HSV color detection on your hand
  3. Show if your hand color matches the HSV range
  4. Suggest next steps
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def check_camera():
    """Test if camera can be opened."""
    print("\n" + "="*60)
    print("1. CHECKING CAMERA...")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera device 0")
        print("\nTroubleshooting:")
        print("  • Is a webcam plugged in?")
        print("  • Is another app using the camera (Zoom, Skype, etc)?")
        print("  • Try plugging in USB camera and restart")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot read frames from camera")
        cap.release()
        return None
    
    print("✅ Camera OK - Reading frames successfully")
    return cap

def test_hand_detection(cap):
    """Test if hand is detected with default HSV range."""
    print("\n" + "="*60)
    print("2. TESTING HAND DETECTION...")
    print("="*60)
    print("\nInstructions:")
    print("  1. Show your HAND to the camera")
    print("  2. Hold it steady")
    print("  3. Watch the right panel (should turn GREEN if hand detected)")
    print("  4. Press 'q' to continue")
    print("\nWhat you should see:")
    print("  LEFT panel:   Your hand (original camera feed)")
    print("  RIGHT panel:  Green areas = detected skin (bad/no green = HSV mismatch)")
    
    detected_any = False
    frames_checked = 0
    max_pixels_detected = 0
    
    print("\nRunning 100 frames of analysis...\n")
    
    try:
        while frames_checked < 100:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Apply default HSV range
            lower_skin = np.array([0, 30, 80], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            pixels_detected = np.count_nonzero(mask)
            max_pixels_detected = max(max_pixels_detected, pixels_detected)
            
            if pixels_detected > 100:
                detected_any = True
            
            # Display side-by-side
            display = np.hstack([frame, mask_display])
            
            # Add text overlay
            text = f"Frames: {frames_checked+1}/100 | Pixels detected: {pixels_detected}"
            cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if pixels_detected > 100:
                cv2.putText(display, "✓ HAND DETECTED!", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(display, "✗ No hand (green areas not visible?)", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Quick Hand Check - Press 'q' to continue", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frames_checked += 1
        
        cv2.destroyAllWindows()
        
        print(f"\n✅ Analysis complete ({frames_checked} frames)")
        print(f"   Max pixels detected: {max_pixels_detected}")
        print(f"   Hand detection: {'YES ✓' if detected_any else 'NO ✗'}")
        
        return detected_any, max_pixels_detected
        
    except Exception as e:
        print(f"❌ Error during detection test: {e}")
        return False, 0

def check_hsv_statistics(cap):
    """Check HSV values of hand area."""
    print("\n" + "="*60)
    print("3. ANALYZING YOUR HAND COLOR...")
    print("="*60)
    print("\nShow your hand in the CENTER of the frame")
    print("Analyzing HSV values for 30 frames...\n")
    
    h_values = []
    s_values = []
    v_values = []
    
    try:
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get center region
            h, w = frame.shape[:2]
            roi = frame[h//2-30:h//2+30, w//2-30:w//2+30]
            
            if roi.size == 0:
                continue
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            h_values.extend(hsv[:,:,0].flatten())
            s_values.extend(hsv[:,:,1].flatten())
            v_values.extend(hsv[:,:,2].flatten())
            
            print(".", end="", flush=True)
        
        print("\n\n📊 Your Hand Color Statistics:")
        print(f"  Hue range:        {int(np.min(h_values))}-{int(np.max(h_values))} " +
              f"(default: 0-20) {'✓' if np.max(h_values) <= 20 else '⚠'}")
        print(f"  Saturation range: {int(np.min(s_values))}-{int(np.max(s_values))} " +
              f"(default: 30-150) {'✓' if np.min(s_values) >= 25 else '⚠'}")
        print(f"  Value range:      {int(np.min(v_values))}-{int(np.max(v_values))} " +
              f"(default: 80-255) {'✓' if np.min(v_values) >= 70 else '⚠'}")
        
        return h_values, s_values, v_values
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return [], [], []

def print_recommendations(detected, pixels, h_vals, s_vals, v_vals):
    """Print actionable recommendations."""
    print("\n" + "="*60)
    print("RESULTS & RECOMMENDATIONS")
    print("="*60)
    
    if detected and pixels > 5000:
        print("\n✅ GREAT NEWS! Hand detection is working!")
        print("   Your hand matches the default HSV range")
        print("\n🚀 Next steps:")
        print("   1. Run: python src/main.py")
        print("   2. Show your hand to the camera")
        print("   3. Open and close your hand to control gestures")
        
    elif detected and pixels > 1000:
        print("\n⚠️  Hand detection partially working")
        print("   Your hand is detected but with low confidence")
        print("\n   Possible issues:")
        print("   • Poor lighting")
        print("   • Hand too far from camera")
        print("   • Hand partially out of frame")
        
        if h_vals:
            h_max = int(np.max(h_vals))
            if h_max > 20:
                print(f"\n   Your Hue reaches {h_max} (default max: 20)")
                print("   Try running: python diagnose_hand_detection.py")
    
    else:
        print("\n❌ Hand detection NOT working")
        print("   Your hand is not matching the HSV color range")
        
        if h_vals:
            print("\n   Your hand colors:")
            h_mid = int(np.mean(h_vals))
            s_mid = int(np.mean(s_vals))
            v_mid = int(np.mean(v_vals))
            print(f"   H: {int(np.min(h_vals))}-{int(np.max(h_vals))} (avg: {h_mid})")
            print(f"   S: {int(np.min(s_vals))}-{int(np.max(s_vals))} (avg: {s_mid})")
            print(f"   V: {int(np.min(v_vals))}-{int(np.max(v_vals))} (avg: {v_mid})")
            
            print("\n   🔧 FIX: Run the diagnostic tool:")
            print("      python src/diagnose_hand_detection.py")
            print("\n      This will:")
            print("      • Show your hand in green if color matches")
            print("      • Suggest optimal HSV range for YOUR skin tone")
            print("      • Help you configure detection for your setup")
        
        print("\n   Or try these quick fixes:")
        print("   1. Improve lighting (use a desk lamp)")
        print("   2. Move hand closer to camera (20-40 cm)")
        print("   3. Make sure hand is fully in frame")
    
    print("\n📚 For more help: Read HAND_DETECTION_TROUBLESHOOTING.md")
    print("="*60 + "\n")

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("HAND GESTURE CONTROLLER - QUICK CHECK")
    print("="*60)
    
    # Check 1: Camera
    cap = check_camera()
    if cap is None:
        print("\n❌ Cannot proceed without camera")
        sys.exit(1)
    
    # Check 2: Hand detection
    detected, pixels = test_hand_detection(cap)
    
    # Check 3: HSV analysis
    h_vals, s_vals, v_vals = check_hsv_statistics(cap)
    
    # Recommendations
    print_recommendations(detected, pixels, h_vals, s_vals, v_vals)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
