from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time

cv2.namedWindow("Pick 4 court corners (bottom-L, bottom-R, top-R, top-L)", cv2.WINDOW_NORMAL)

REAL_TIME = True

# Video path for testing
if REAL_TIME:
    video = 0  # Use webcam
else:
    video = r"C:\Users\victo\OneDrive\Documents\Cambridge\Computing\Badminton tracker\matchNew.mp4"

LEFT_ANKLE   = 15
RIGHT_ANKLE  = 16
LEFT_HIP     = 11
RIGHT_HIP    = 12
TRAIL_LENGTH  = 30 # how many past positions to show in the trail
MIN_MOVEMENT  = 5 # Filter out any small jitter
MOVEMENT_WINDOW = 20 # how many frames of history to consider for movement detection
MEMORY_FRAMES = 15
VELOCITY_WINDOW = 5
MIN_CONTACT_GAP = 3

# Recovery speed tuning NEED TO TUNE
LUNGE_THRESHOLD  = 120   # pixels from centre — ankle this far out = lunge
RECOVERY_TIMEOUT = 60    # frames to measure recovery over

# COURT SELECTION ..........................
points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(img, str(len(points)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 255), 2)
        if len(points) == 4:
            cv2.line(img, points[-1], points[0], (0, 255, 255), 2)
            cv2.putText(img, "Press any key to start tracking",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pick 4 court corners (bottom-L, bottom-R, top-R, top-L)", img)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ret, img = cap.read()
frame_h, frame_w = img.shape[:2]
cap.release()

cv2.imshow("Pick 4 court corners (bottom-L, bottom-R, top-R, top-L)", img)
cv2.setMouseCallback("Pick 4 court corners (bottom-L, bottom-R, top-R, top-L)", click)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow("Foot Tracking + Metrics", cv2.WINDOW_NORMAL)
cv2.namedWindow("Final Heatmap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Foot Tracking + Metrics", 1280, 720)

if len(points) < 4:
    print("Need 4 points — exiting.")
    exit()

COURT_POLYGON = np.array(points, dtype=np.int32)

# HOMOGRAPHY SETUP ...........................

# Real badminton court dimensions in pixels for top-down view
# Standard court is 13.4m x 6.1m, we scale to pixels
TOP_DOWN_W = 400
TOP_DOWN_H = 880  

# Destination points for top-down view (corners of the rectangle)
dst_points = np.array([
    [0,          TOP_DOWN_H],   # bottom-left
    [TOP_DOWN_W, TOP_DOWN_H],   # bottom-right
    [TOP_DOWN_W, 0          ],  # top-right
    [0,          0          ],  # top-left
], dtype=np.float32)

src_points = np.array(points, dtype=np.float32)
H, _ = cv2.findHomography(src_points, dst_points)

# Separate window for top-down view
cv2.namedWindow("Top-Down Heatmap", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Top-Down Heatmap", TOP_DOWN_W, TOP_DOWN_H)

# Top-down heatmap accumulator
heatmap_td = np.zeros((TOP_DOWN_H, TOP_DOWN_W), dtype=np.float32)

# Maybe needed not sure for players jumping off court
def expand_polygon_upward(polygon, pixels=200):
    expanded = polygon.copy().astype(np.float32)
    top_indices = np.argsort(expanded[:, 1])[:2]
    expanded[top_indices, 1] -= pixels
    return expanded.astype(np.int32)

TRACKING_POLYGON = expand_polygon_upward(COURT_POLYGON, pixels=200)

def is_on_court(x, y):
    return cv2.pointPolygonTest(TRACKING_POLYGON, (float(x), float(y)), False) >= 0

# Movement detection to remove noise
def is_moving(position_history):
    if len(position_history) < 2:
        return False
    xs = [p[0] for p in position_history]
    ys = [p[1] for p in position_history]
    return (max(xs) - min(xs) + max(ys) - min(ys)) > MIN_MOVEMENT

def detect_ground_contact(y_history, frames_since_last):
    if len(y_history) < VELOCITY_WINDOW * 2 + 1 or frames_since_last < MIN_CONTACT_GAP:
        return False
    ys = list(y_history)
    # Remove any jitter from pose estimation
    smoothed = np.convolve(ys, np.ones(VELOCITY_WINDOW) / VELOCITY_WINDOW, mode='valid')
    if len(smoothed) < 3:
        return False
    mid = len(smoothed) - 2
    # Contact if we have a local max in the smoothed y (foot lowest) and it's near the bottom of recent movement
    is_local_max = smoothed[mid] > smoothed[mid-1] and smoothed[mid] > smoothed[mid+1]
    y_range = max(smoothed) - min(smoothed)
    near_bottom = smoothed[mid] > (min(smoothed) + y_range * 0.6)
    return is_local_max and near_bottom

def draw_metric_panel(frame, player_idx, metrics, x_offset, y_offset):
    """Draw a stats panel for a player."""
    color   = (0, 255, 0) if player_idx == 0 else (0, 100, 255)
    label   = f"Player {player_idx + 1}"
    padding = 10
    panel_w = 280
    panel_h = 110

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset),
                  (x_offset + panel_w, y_offset + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label,
                (x_offset + padding, y_offset + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Step freq:  {metrics['step_freq']:.2f} steps/s",
                (x_offset + padding, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Contacts:   {metrics['total_contacts']}",
                (x_offset + padding, y_offset + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Recovery:   {metrics['recovery']}",
                (x_offset + padding, y_offset + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ── Step 2: track ───────────────────────────────────────────────────────────
model = YOLO("yolov8n-pose.pt")
cap   = cv2.VideoCapture(video)

player_trails        = {}
on_court_memory      = {}
position_history     = {}
y_history            = {}
frames_since_contact = {}

# Metrics per player
contact_timestamps   = {}   # {player_idx: [frame_numbers of contacts]}
lunge_state          = {}   # {player_idx: {'in_lunge': bool, 'lunge_frame': int, 'lunge_pos': (x,y)}}
recovery_speeds      = {}   # {player_idx: [frames to recover]}

heatmap    = np.zeros((frame_h, frame_w), dtype=np.float32)
frame_idx  = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame, verbose=False)

    cv2.polylines(frame, [COURT_POLYGON],    isClosed=True, color=(255, 255, 0), thickness=2)
    cv2.polylines(frame, [TRACKING_POLYGON], isClosed=True, color=(255, 100, 0), thickness=1)

    seen_players = set()

    for result in results:
        if result.keypoints is None:
            continue

        for player_idx, person in enumerate(result.keypoints.xy):
            seen_players.add(player_idx)

            lax, lay = int(person[LEFT_ANKLE][0]),  int(person[LEFT_ANKLE][1])
            rax, ray = int(person[RIGHT_ANKLE][0]), int(person[RIGHT_ANKLE][1])
            lhx, lhy = int(person[LEFT_HIP][0]),    int(person[LEFT_HIP][1])
            rhx, rhy = int(person[RIGHT_HIP][0]),   int(person[RIGHT_HIP][1])

            if is_on_court(lax, lay) or is_on_court(rax, ray):
                on_court_memory[player_idx] = 0
            elif player_idx in on_court_memory:
                on_court_memory[player_idx] += 1

            if on_court_memory.get(player_idx, MEMORY_FRAMES + 1) > MEMORY_FRAMES:
                continue

            # Init structures
            if player_idx not in position_history:
                position_history[player_idx]     = {LEFT_ANKLE:  deque(maxlen=MOVEMENT_WINDOW),
                                                     RIGHT_ANKLE: deque(maxlen=MOVEMENT_WINDOW)}
                y_history[player_idx]            = {LEFT_ANKLE:  deque(maxlen=VELOCITY_WINDOW * 3),
                                                     RIGHT_ANKLE: deque(maxlen=VELOCITY_WINDOW * 3)}
                frames_since_contact[player_idx] = {LEFT_ANKLE: 999, RIGHT_ANKLE: 999}
                player_trails[player_idx]        = {LEFT_ANKLE: [], RIGHT_ANKLE: []}
                contact_timestamps[player_idx]   = []
                lunge_state[player_idx]          = {'in_lunge': False, 'lunge_frame': 0, 'lunge_pos': None}
                recovery_speeds[player_idx]      = []

            # Hip centre (used for lunge detection)
            hip_cx = (lhx + rhx) // 2
            hip_cy = (lhy + rhy) // 2

            for fid, color in [(LEFT_ANKLE, (0, 255, 0)), (RIGHT_ANKLE, (0, 100, 255))]:
                x, y = int(person[fid][0]), int(person[fid][1])
                if x == 0 and y == 0:
                    continue

                position_history[player_idx][fid].append((x, y))
                y_history[player_idx][fid].append(y)
                frames_since_contact[player_idx][fid] += 1

                if not is_moving(position_history[player_idx][fid]):
                    continue

                # Trail
                trail = player_trails[player_idx][fid]
                if trail:
                    lx2, ly2 = trail[-1]
                    if abs(x - lx2) > 200 or abs(y - ly2) > 200:
                        trail.clear()
                trail.append((x, y))
                # Clears if jump is too large (teleport detection)
                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i-1], trail[i], color,
                             max(1, int(3 * i / len(trail))))

                cv2.circle(frame, (x, y), 8, color, -1)

                # Ground contact
                contact = detect_ground_contact(
                    y_history[player_idx][fid],
                    frames_since_contact[player_idx][fid]
                )
                if contact and is_on_court(x, y):
                    frames_since_contact[player_idx][fid] = 0
                    contact_timestamps[player_idx].append(frame_idx)
                    cv2.circle(heatmap, (x, y), 25, 1.0, -1)
                    cv2.circle(frame, (x, y), 18, (255, 255, 255), 3)

                    # Transform point to top-down coordinates
                    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
                    td_pt = cv2.perspectiveTransform(pt, H)[0][0]
                    tdx, tdy = int(td_pt[0]), int(td_pt[1])
                    if 0 <= tdx < TOP_DOWN_W and 0 <= tdy < TOP_DOWN_H:
                        cv2.circle(heatmap_td, (tdx, tdy), 20, 1.0, -1)

                    # Lunge detection — ankle far from hip centre
                    dist_from_hip = np.hypot(x - hip_cx, y - hip_cy)
                    ls = lunge_state[player_idx]

                    if dist_from_hip > LUNGE_THRESHOLD and not ls['in_lunge']:
                        ls['in_lunge']    = True
                        ls['lunge_frame'] = frame_idx
                        ls['lunge_pos']   = (x, y)

                    elif dist_from_hip < LUNGE_THRESHOLD * 0.6 and ls['in_lunge']:
                        # Recovered — measure frames taken
                        recovery_frames = frame_idx - ls['lunge_frame']
                        recovery_speeds[player_idx].append(recovery_frames)
                        ls['in_lunge'] = False

            # Top-down heatmap display
            td_display = np.zeros((TOP_DOWN_H, TOP_DOWN_W, 3), dtype=np.uint8)

            # Draw court lines on top-down view
            court_lines = [
                # Outer boundary
                (0, 0), (TOP_DOWN_W, 0), (TOP_DOWN_W, TOP_DOWN_H), (0, TOP_DOWN_H),
            ]
            cv2.rectangle(td_display, (2, 2), (TOP_DOWN_W-2, TOP_DOWN_H-2), (255, 255, 255), 2)
            # Net line
            cv2.line(td_display, (0, TOP_DOWN_H//2), (TOP_DOWN_W, TOP_DOWN_H//2), (255, 255, 255), 2)
            # Service lines (roughly 1.98m from net = ~13% of court length)
            service_offset = int(TOP_DOWN_H * 0.148)
            cv2.line(td_display, (0, TOP_DOWN_H//2 - service_offset),
                    (TOP_DOWN_W, TOP_DOWN_H//2 - service_offset), (200, 200, 200), 1)
            cv2.line(td_display, (0, TOP_DOWN_H//2 + service_offset),
                    (TOP_DOWN_W, TOP_DOWN_H//2 + service_offset), (200, 200, 200), 1)
            # Centre line
            cv2.line(td_display, (TOP_DOWN_W//2, TOP_DOWN_H//2 - service_offset),
                    (TOP_DOWN_W//2, TOP_DOWN_H//2 + service_offset), (200, 200, 200), 1)

            if heatmap_td.max() > 0:
                heatmap_td_norm  = cv2.normalize(heatmap_td, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_td_color = cv2.applyColorMap(heatmap_td_norm.astype(np.uint8), cv2.COLORMAP_JET)
                # Mask to court boundary
                td_mask = np.zeros((TOP_DOWN_H, TOP_DOWN_W), dtype=np.uint8)
                cv2.rectangle(td_mask, (0, 0), (TOP_DOWN_W, TOP_DOWN_H), 255, -1)
                heatmap_td_color[td_mask == 0] = 0
                td_display = cv2.addWeighted(td_display, 0.4, heatmap_td_color, 0.6, 0)
                # Redraw court lines on top of heatmap
                cv2.rectangle(td_display, (2, 2), (TOP_DOWN_W-2, TOP_DOWN_H-2), (255, 255, 255), 2)
                cv2.line(td_display, (0, TOP_DOWN_H//2), (TOP_DOWN_W, TOP_DOWN_H//2), (255, 255, 255), 2)
                cv2.line(td_display, (0, TOP_DOWN_H//2 - service_offset),
                        (TOP_DOWN_W, TOP_DOWN_H//2 - service_offset), (200, 200, 200), 1)
                cv2.line(td_display, (0, TOP_DOWN_H//2 + service_offset),
                        (TOP_DOWN_W, TOP_DOWN_H//2 + service_offset), (200, 200, 200), 1)
                cv2.line(td_display, (TOP_DOWN_W//2, TOP_DOWN_H//2 - service_offset),
                        (TOP_DOWN_W//2, TOP_DOWN_H//2 + service_offset), (200, 200, 200), 1)

            cv2.imshow("Top-Down Heatmap", td_display)

            # Draw lunge indicator
            if lunge_state[player_idx]['in_lunge']:
                cv2.putText(frame, "LUNGE",
                            (hip_cx - 30, hip_cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Compute and display metrics
        for player_idx in list(on_court_memory.keys()):
            if on_court_memory[player_idx] > MEMORY_FRAMES:
                continue

            # Step frequency over last 3 seconds
            window_frames  = int(fps * 3)
            recent_contacts = [t for t in contact_timestamps.get(player_idx, [])
                               if frame_idx - t < window_frames]
            step_freq = len(recent_contacts) / 3.0

            # Average recovery speed in frames → seconds
            rec_list = recovery_speeds.get(player_idx, [])
            if rec_list:
                avg_rec = np.mean(rec_list[-5:])  # last 5 lunges
                rec_str = f"{avg_rec / fps:.2f}s avg"
            else:
                rec_str = "no lunges yet"

            metrics = {
                'step_freq':      step_freq,
                'total_contacts': len(contact_timestamps.get(player_idx, [])),
                'recovery':       rec_str,
            }

            x_off = 10 + player_idx * 300
            draw_metric_panel(frame, player_idx, metrics, x_off, 10)

    for pid in list(on_court_memory.keys()):
        if pid not in seen_players:
            on_court_memory[pid] += 1
            if on_court_memory[pid] > MEMORY_FRAMES:
                del on_court_memory[pid]

    # Heatmap overlay — only show once we have contacts
    if heatmap.max() > 0:
        heatmap_norm  = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [COURT_POLYGON], 255)
        heatmap_color[mask == 0] = 0
        frame = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)


    
    
    cv2.imshow("Foot Tracking + Metrics", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()

# ── Step 3: print summary + save heatmap ────────────────────────────────────
print("\n=== MATCH SUMMARY ===")
for pid in sorted(contact_timestamps.keys()):
    total   = len(contact_timestamps[pid])
    elapsed = frame_idx / fps
    freq    = total / elapsed if elapsed > 0 else 0
    recs    = recovery_speeds.get(pid, [])
    avg_rec = (np.mean(recs) / fps) if recs else 0
    print(f"Player {pid + 1}: {total} contacts | "
          f"{freq:.2f} steps/s avg | "
          f"{avg_rec:.2f}s avg recovery ({len(recs)} lunges)")

cap = cv2.VideoCapture(video)
ret, bg = cap.read()
cap.release()

heatmap_norm  = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
cv2.fillPoly(mask, [COURT_POLYGON], 255)
heatmap_color[mask == 0] = 0
final = cv2.addWeighted(bg, 0.5, heatmap_color, 0.5, 0)
cv2.imwrite("heatmap_final.png", final)
cv2.imwrite("heatmap_topdown.png", td_display)
print("Saved heatmap_final.png")

cv2.imshow("Final Heatmap", final)
cv2.waitKey(0)
cv2.destroyAllWindows()