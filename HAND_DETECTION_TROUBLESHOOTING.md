# HAND DETECTION TROUBLESHOOTING GUIDE

## Problem: Hand is Not Being Detected

If the system is not detecting your hand, don't worry! This guide will help you diagnose and fix the issue.

---

## 📊 STEP 1: Check Detection Logs

When you run the application, it automatically creates detailed logs of every detection attempt.

### Location of Log Files
```
e:\Project\Hand Gesture Controller\logs\hand_detection_YYYYMMDD_HHMMSS.log
```

### What to Look For in Logs

**✓ Good Detection (should see a line like this):**
```
14:32:15 - [INFO] - Frame 42: ✓ DETECTED
  → Hand contour area: 25430.0 pixels
  → Bounding box: x=120, y=95, w=180, h=220
  → Centroid: (210.5, 205.3)
  → Detection rate: 85.7%
```

**✗ Failed Detection (should see a line like this):**
```
14:32:16 - [WARNING] - Frame 43: ✗ NOT DETECTED
  → Reason: No valid hand contour found
  → HSV mask pixels: 15240 (8.1%)
  → After cleaning: 3200 (1.7%)
  → Detection rate: 84.6%
```

### How to Read the Logs

- **"Frame X: ✓ DETECTED"** = Hand successfully found in that frame
- **"Frame X: ✗ NOT DETECTED"** = Hand NOT found - system is trying to find it
  - Check: "HSV mask pixels" - this shows how many pixels passed the color filter
  - If this is very low (<5%), your hand color isn't matching the HSV range

---

## 🔧 STEP 2: Run the Diagnostic Tool

The application includes an interactive diagnostic tool to analyze why detection is failing.

```bash
cd e:\Project\Hand Gesture Controller\src
python diagnose_hand_detection.py
```

### What the Diagnostic Tool Does

1. **Opens your webcam** and shows real-time HSV analysis
2. **Displays three views:**
   - LEFT: Your hand in the camera
   - MIDDLE: Which pixels are detected as skin color (green = detected)
   - RIGHT: After cleaning up noise

3. **Guides you through**:
   - Showing your hand from different angles
   - Different lighting conditions
   - Different distances from camera

4. **Analyzes the results** and recommends HSV range adjustments

### How to Use the Diagnostic Tool

**Step 1:** Start the tool
```bash
python diagnose_hand_detection.py
```

**Step 2:** When you see the three video panels:
- Position your hand in frame
- Look at the MIDDLE panel to see if your hand turns green
- If the middle panel stays black → HSV range doesn't match your skin

**Step 3:** Capture frames
- Press SPACE 4-5 times with your hand in different positions
- Try different lighting conditions
- Try different distances (20cm, 40cm, 60cm away)
- Press 'q' when done

**Step 4:** View recommendations
- The tool will print suggested HSV values
- It will explain what needs to be adjusted

---

## 🎯 COMMON ISSUES & FIXES

### Issue 1: "Mask pixels: 0% - Never" (HSV range doesn't match your hand)

**Symptom:** Middle panel stays completely black, no green pixels even when hand is clearly visible

**Cause:** Your skin tone is outside the default HSV range

**Fix:**
1. Run the diagnostic tool and capture frames
2. It will suggest a new HSV range
3. In `main.py`, modify line ~85:
```python
self.detector = HandDetectionSegmentor(
    hsv_lower=(0, 30, 80),      # Changed lower bounds
    hsv_upper=(20, 150, 255),   # Changed upper bounds
)
```

**Example:** If diagnostic suggests `hsv_lower=(5, 25, 70)` and `hsv_upper=(25, 160, 255)`:
```python
self.detector = HandDetectionSegmentor(
    hsv_lower=(5, 25, 70),
    hsv_upper=(25, 160, 255),
)
```

---

### Issue 2: "Lighting conditions too extreme" (Too bright or too dark)

**Symptom:** Mask shows some pixels but they're very fragmented after cleaning

**Cause:** Lighting is too inconsistent

**Fix:**
1. **Best solution:** Improve lighting
   - Use a desk lamp or window light
   - Avoid direct shadows on hand
   - Not too bright (causes glare)
   - Not too dark (causes underexposure)

2. **If you can't improve lighting:**
   - Adjust Saturation range in HSV
   - Try lowering `hsv_lower = (0, 20, 50)` instead of `(0, 30, 80)`

---

### Issue 3: "False positives" (Detecting background, not just hand)

**Symptom:** Green detection box appears on walls, clothing, or other objects

**Cause:** HSV range is too loose

**Fix:** Tighten HSV range
```python
self.detector = HandDetectionSegmentor(
    hsv_lower=(0, 40, 100),    # Higher saturation threshold
    hsv_upper=(15, 140, 245),  # Lower hue upper bound
)
```

---

### Issue 4: "Hand detected but gestures don't register"

**Symptom:** Debug overlay shows "✓ No hand detected" but hand is visible

**Cause:** Hand area passes HSV test but fails minimum area or solidity requirements

**Fix:** Check if hand is too small
```python
self.detector = HandDetectionSegmentor(
    min_contour_area=3000,      # Reduced from 5000
    min_contour_solidity=0.5,   # Reduced from 0.6
)
```

---

### Issue 5: "Webcam problem" (Can't open camera)

**Symptom:** Error message "Failed to open camera device"

**Cause:** Wrong camera index or camera already in use

**Fix:**
```python
# Try different camera indices (0, 1, 2, etc.)
self.preprocessor = VideoCapturePreprocessor(
    device_index=1,  # Change to 1, 2, 3 if 0 doesn't work
)
```

Or check if another application is using the camera:
- Close Zoom, Skype, OBS, etc.
- Restart the application

---

## 📈 MONITORING DETECTION PERFORMANCE

### During Runtime

Press **'s'** key to print live statistics:

```
✅ HAND DETECTION STATUS:
  Total frames: 150
  Detections: 127
  Detection rate: 84.7%
  Last frame: ✓ Hand detected
```

**Good metrics:**
- Detection rate > 80%
- Consistent detections across frames

**Bad metrics:**
- Detection rate < 50%
- Intermittent detections (works then stops)

### After Session

Open the log file in the `logs/` folder:
```
e:\Project\Hand Gesture Controller\logs\hand_detection_20250309_143215.log
```

Each line shows what happened in that frame:
- Frame number
- Detection success/failure
- HSV statistics
- Why it failed (if it failed)

---

## 🛠️ ADVANCED TROUBLESHOOTING

### Problem: Hand Detected but Finger Count Wrong

**Check:** Feature extraction logs (Module 3)

The system uses convex hull analysis. Edges matter:
- Fingers must be separated
- Angles must be clear
- This is geometric, not ML-based

**Fix:** 
- Open/close hand fully (all fingers extended or fully closed)
- Keep hand aligned with camera
- Avoid overlapping fingers

### Problem: Frame Rate Drops Below 20 FPS

**Check:** HSV range size

Large HSV ranges process more pixels. If `hsv_upper - hsv_lower` is very large, it slows down processing.

**Fix:**
- Tighten HSV range to match your specific skin tone
- Don't use `hsv_lower=(0, 0, 0)` and `hsv_upper=(180, 255, 255)` (the entire spectrum)

---

## 📝 DIAGNOSIS CHECKLIST

If hand detection isn't working, go through this checklist:

```
□ Hand is clearly visible in camera view
□ Hand is not too far away (20-60 cm from camera)
□ Lighting is adequate (not too dark, not too bright)
□ No shadows on hand
□ Hand fills at least 10% of the frame
□ HSV range matches your skin tone (use diagnostic tool)
□ Only hand is in frame (no other skin-colored objects)
□ Camera is not already in use by another app
□ Sufficient disk space for logs
□ Latest OpenCV installed (4.5 or newer)
```

---

## 🚀 QUICK START FOR COMMON SCENARIOS

### Scenario 1: "Dark skin tone not detecting"
```python
# Use wider saturation range
self.detector = HandDetectionSegmentor(
    hsv_lower=(0, 30, 60),      # Lower V threshold
    hsv_upper=(25, 160, 255),   # Same saturation
)
```

### Scenario 2: "Fair/pale skin not detecting"
```python
# Use narrower saturation range
self.detector = HandDetectionSegmentor(
    hsv_lower=(0, 10, 80),      # Lower saturation
    hsv_upper=(20, 120, 255),
)
```

### Scenario 3: "Hand too small in frame"
```python
# Lower minimum area requirement
self.detector = HandDetectionSegmentor(
    min_contour_area=2000,      # Instead of 5000
)
```

---

## 📞 IF STILL NOT WORKING

1. **Check the log file** - it tells you exactly what's happening each frame
2. **Run the diagnostic tool** - it analyzes your specific hand/lighting/camera
3. **Review HSV range** - this is 90% of detection issues
4. **Verify camera works** - test with other applications (e.g., Zoom)
5. **Check lighting** - take a photo with phone camera to verify what camera sees

---

## 📊 EXPECTED PERFORMANCE

With proper setup:
- **Detection rate:** 85-95% (hand visible in frame)
- **Gesture accuracy:** 90% (correct finger count)
- **Frame rate:** 25-30 FPS (CPU standard laptop)
- **Latency:** 40-70 ms (gesture to game action)

---

## 🔍 UNDERSTANDING THE DEBUG DISPLAY

When you run the application with debug enabled (press 'd' to toggle):

```
Top-left: Gesture name and confidence
Top-right: Detection rate (e.g., "Detection: 85%")
Center-left: Shows hand contour (green) and bounding box (blue)
Center-right: Red dots = fingertips
Bottom: FPS and frame counter
```

If you see "No hand detected" in red text - the HSV range didn't find any skin pixels.

---

## Key Files for Configuration

- **Hand Detection Module:** `src/modules/hand_detection_segmentation.py`
  - HSV range: lines 40-50
  - Minimum area/solidity: lines 40-50
  - Morphological operations: lines 80-100

- **Main Application:** `src/main.py`
  - Where detector is initialized: line ~85

- **Diagnostic Tool:** `src/diagnose_hand_detection.py`
  - Run this to find your optimal HSV range

---

Last updated: March 2025
For more help: Check logs in `e:\Project\Hand Gesture Controller\logs\`
