import cv2
import numpy as np

cap = cv2.VideoCapture("differentMatch.mp4")
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)

# Resize the video to 1080p for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # 2. Clean up motion mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # 3. White colour mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    # 4. Combine motion + white
    combined = cv2.bitwise_and(fg_mask, white_mask)

    # 5. First pass contours — blank out players
    contours_first, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_first:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            combined[y:y+h, x:x+w] = 0

    # 6. Second pass contours — shuttle candidates only
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Filter candidates
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (3 < area < 400):      # wider range to handle different zoom levels
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 60 or h > 60:              # max size to exclude players
            continue
        if w < 5 or h < 5:              # min size to exclude noise
            continue
        aspect = w / max(h, 1)
        if not (0.5 < aspect < 2.0):
            continue
        candidates.append((area, x, y, w, h))

    shuttle_pos = None
    if candidates:
        candidates.sort(key=lambda c: c[0])
        _, x, y, w, h = candidates[0]
        shuttle_pos = (x, y, w, h)

    # 8. Draw box on full-res frame
    if shuttle_pos:
        x, y, w, h = shuttle_pos
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "shuttle", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 9. Resize for display only
    display_size = (1280, 720)
    cv2.imshow("Shuttle Tracker", cv2.resize(frame, display_size))
    cv2.imshow("FG Mask",         cv2.resize(fg_mask, display_size))
    cv2.imshow("White Mask",      cv2.resize(white_mask, display_size))
    cv2.imshow("Combined",        cv2.resize(combined, display_size))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Shuttle Tracker", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()