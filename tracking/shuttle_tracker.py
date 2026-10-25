import cv2
import numpy as np

cap = cv2.VideoCapture("matchTrimmed.mp4")

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

last_pos = None
last_vel = (0, 0)        # velocity from last two frames
trail = []
TRAIL_LENGTH = 20
MAX_JUMP = 250
lost_count = 0           # frames since shuttle was last seen
MAX_LOST = 10            # reset tracking after this many missed frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Frame difference
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 2. White colour mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))

    # 3. Combine
    combined = cv2.bitwise_and(motion_mask, white_mask)

    # 4. Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # 5. Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Filter to shuttle-sized candidates
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (4 < area < 120):     # tighter upper bound — shirts/shoes are bigger
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 35 or h > 35:
            continue
        aspect = w / max(h, 1)
        if not (0.3 < aspect < 3.0):
            continue
        cx, cy = x + w // 2, y + h // 2

        # When we have no tracking history, ignore the bottom 30% of frame (shoes)
        # and the mid-height player body region
        if last_pos is None:
            if cy > h_frame * 0.70:
                continue

        candidates.append((area, cx, cy, x, y, w, h))

    # 7. Pick best candidate using scoring
    shuttle_pos = None
    if candidates:
        if last_pos is None:
            # No history — pick smallest blob in upper portion
            candidates.sort(key=lambda c: c[0])
            shuttle_pos = candidates[0]
        else:
            lx, ly = last_pos
            vx, vy = last_vel

            # Predicted position based on last velocity
            pred_x = lx + vx
            pred_y = ly + vy

            best_score = float('inf')
            best = None
            for cand in candidates:
                area, cx, cy, x, y, w, h = cand

                # Distance from predicted position
                dist = np.hypot(cx - pred_x, cy - pred_y)
                if dist > MAX_JUMP:
                    continue

                # Small size bonus — shuttle is smaller than shoes/shirt
                size_penalty = area * 1.5

                score = dist + size_penalty
                if score < best_score:
                    best_score = score
                    best = cand

            shuttle_pos = best

    # 8. Update tracking state
    if shuttle_pos:
        area, cx, cy, x, y, w, h = shuttle_pos
        if last_pos:
            last_vel = (cx - last_pos[0], cy - last_pos[1])
        last_pos = (cx, cy)
        lost_count = 0
        trail.append((cx, cy))
        if len(trail) > TRAIL_LENGTH:
            trail.pop(0)
    else:
        lost_count += 1
        if lost_count > MAX_LOST:
            last_pos = None
            last_vel = (0, 0)
        if trail:
            trail.pop(0)

    # 9. Draw trail
    display = frame.copy()
    for i, (tx, ty) in enumerate(trail):
        alpha = (i + 1) / len(trail)
        radius = max(2, int(4 * alpha))
        color_intensity = int(255 * alpha)
        cv2.circle(display, (tx, ty), radius,
                   (0, color_intensity, 255 - color_intensity), -1)

    # 10. Draw bounding box
    if shuttle_pos:
        area, cx, cy, x, y, w, h = shuttle_pos
        pad = 6
        cv2.rectangle(display,
                      (x - pad, y - pad),
                      (x + w + pad, y + h + pad),
                      (0, 255, 0), 2)
        cv2.putText(display, "shuttle", (x - pad, y - pad - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 11. Show lost count for debugging
    status = f"Tracking" if last_pos else f"Searching... (lost {lost_count}f)"
    cv2.putText(display, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if last_pos else (0, 0, 255), 2)

    cv2.imshow("Shuttle Tracker", cv2.resize(display, (1280, 720)))
    cv2.imshow("Combined Mask",   cv2.resize(combined, (1280, 720)))

    prev_gray = gray

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Shuttle Tracker", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()