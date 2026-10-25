from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

clip = "matchNew.mp4"

LEFT_ANKLE   = 15
RIGHT_ANKLE  = 16
TRAIL_LENGTH  = 30
MIN_MOVEMENT  = 5
MOVEMENT_WINDOW = 20
MEMORY_FRAMES = 15

VELOCITY_WINDOW = 5
MIN_CONTACT_GAP = 3

# ── Badminton court real dimensions (in pixels for the output canvas) ────────
# Standard court: 13.4m long x 6.1m wide. We'll render at 100px per metre.
COURT_M_LONG  = 13.4
COURT_M_WIDE  = 6.1
SCALE         = 80   # pixels per metre
MARGIN        = 60   # pixel margin around court

COURT_PX_H = int(COURT_M_LONG * SCALE)  # top-down height (long axis = vertical)
COURT_PX_W = int(COURT_M_WIDE * SCALE)  # top-down width

OUTPUT_H = COURT_PX_H + 2 * MARGIN
OUTPUT_W = COURT_PX_W + 2 * MARGIN

# The 4 destination corners on the flat court image (matching pick order: BL, BR, TR, TL)
DST_CORNERS = np.array([
    [MARGIN,              OUTPUT_H - MARGIN],   # bottom-left
    [MARGIN + COURT_PX_W, OUTPUT_H - MARGIN],   # bottom-right
    [MARGIN + COURT_PX_W, MARGIN],              # top-right
    [MARGIN,              MARGIN],              # top-left
], dtype=np.float32)

def draw_court_lines(img):
    """Draw standard badminton court lines on a top-down canvas."""
    c = (220, 220, 220)   # line colour
    t = 2                 # thickness

    tl = (MARGIN, MARGIN)
    br = (MARGIN + COURT_PX_W, OUTPUT_H - MARGIN)

    # Outer boundary
    cv2.rectangle(img, tl, br, c, t)

    # Centre line (net)
    mid_y = MARGIN + COURT_PX_H // 2
    cv2.line(img, (MARGIN, mid_y), (MARGIN + COURT_PX_W, mid_y), c, t)

    def ym(metres):
        return MARGIN + int(metres * SCALE)

    def xm(metres):
        return MARGIN + int(metres * SCALE)

    # Short service lines (1.98m from net each side)
    net_y = MARGIN + COURT_PX_H // 2
    svc_offset = int(1.98 * SCALE)
    cv2.line(img, (MARGIN, net_y - svc_offset), (MARGIN + COURT_PX_W, net_y - svc_offset), c, t)
    cv2.line(img, (MARGIN, net_y + svc_offset), (MARGIN + COURT_PX_W, net_y + svc_offset), c, t)

    # Long service line for doubles (0.76m from back)
    back_offset = int(0.76 * SCALE)
    cv2.line(img, (MARGIN, MARGIN + back_offset),              (MARGIN + COURT_PX_W, MARGIN + back_offset), c, t)
    cv2.line(img, (MARGIN, OUTPUT_H - MARGIN - back_offset),   (MARGIN + COURT_PX_W, OUTPUT_H - MARGIN - back_offset), c, t)

    # Centre service line (splits court width in half, between service lines)
    mid_x = MARGIN + COURT_PX_W // 2
    cv2.line(img, (mid_x, net_y - svc_offset), (mid_x, MARGIN + back_offset), c, t)
    cv2.line(img, (mid_x, net_y + svc_offset), (mid_x, OUTPUT_H - MARGIN - back_offset), c, t)

    # Labels
    cv2.putText(img, "NET", (MARGIN + COURT_PX_W // 2 - 20, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)


# ── Step 1: pick court corners ───────────────────────────────────────────────
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

cap = cv2.VideoCapture(clip)
ret, img = cap.read()
frame_h, frame_w = img.shape[:2]
cap.release()

cv2.imshow("Pick 4 court corners (bottom-L, bottom-R, top-R, top-L)", img)
cv2.setMouseCallback("Pick 4 court corners (bottom-L, bottom-R, top-R, top-L)", click)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) < 4:
    print("Need 4 points — exiting.")
    exit()

COURT_POLYGON = np.array(points, dtype=np.int32)
SRC_CORNERS   = np.array(points, dtype=np.float32)

# Compute homography: camera frame → flat top-down court
H, _ = cv2.findHomography(SRC_CORNERS, DST_CORNERS)

def warp_point(x, y):
    """Project a single (x, y) camera point onto the flat court canvas."""
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H)
    return int(warped[0][0][0]), int(warped[0][0][1])

def expand_polygon_upward(polygon, pixels=200):
    expanded = polygon.copy().astype(np.float32)
    top_indices = np.argsort(expanded[:, 1])[:2]
    expanded[top_indices, 1] -= pixels
    return expanded.astype(np.int32)

TRACKING_POLYGON = expand_polygon_upward(COURT_POLYGON, pixels=200)

def is_on_court(x, y):
    return cv2.pointPolygonTest(TRACKING_POLYGON, (float(x), float(y)), False) >= 0

def is_moving(position_history):
    if len(position_history) < 2:
        return False
    xs = [p[0] for p in position_history]
    ys = [p[1] for p in position_history]
    return (max(xs) - min(xs) + max(ys) - min(ys)) > MIN_MOVEMENT

def detect_ground_contact(y_history, frames_since_last_contact):
    if len(y_history) < VELOCITY_WINDOW * 2 + 1:
        return False
    if frames_since_last_contact < MIN_CONTACT_GAP:
        return False
    ys = list(y_history)
    smoothed = np.convolve(ys, np.ones(VELOCITY_WINDOW) / VELOCITY_WINDOW, mode='valid')
    if len(smoothed) < 3:
        return False
    mid = len(smoothed) - 2
    is_local_max = smoothed[mid] > smoothed[mid - 1] and smoothed[mid] > smoothed[mid + 1]
    y_range = max(smoothed) - min(smoothed)
    near_bottom = smoothed[mid] > (min(smoothed) + y_range * 0.6)
    return is_local_max and near_bottom


# ── Step 2: track ────────────────────────────────────────────────────────────
model = YOLO("yolov8n-pose.pt")
cap   = cv2.VideoCapture(clip)

player_trails        = {}
on_court_memory      = {}
position_history     = {}
y_history            = {}
frames_since_contact = {}

# Two separate heatmaps — one per player
heatmap = [
    np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.float32),
    np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.float32),
]

# Colours for players on the live feed
PLAYER_COLORS = [(0, 255, 0), (0, 100, 255)]

# Prepare a blank court canvas for live preview
court_canvas = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
draw_court_lines(court_canvas)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    cv2.polylines(frame, [COURT_POLYGON],    isClosed=True, color=(255, 255, 0), thickness=2)
    cv2.polylines(frame, [TRACKING_POLYGON], isClosed=True, color=(255, 100, 0), thickness=1)

    seen_players = set()
    live_court   = court_canvas.copy()

    for result in results:
        if result.keypoints is None:
            continue

        for player_idx, person in enumerate(result.keypoints.xy):
            seen_players.add(player_idx)

            lax, lay = int(person[LEFT_ANKLE][0]),  int(person[LEFT_ANKLE][1])
            rax, ray = int(person[RIGHT_ANKLE][0]), int(person[RIGHT_ANKLE][1])

            if is_on_court(lax, lay) or is_on_court(rax, ray):
                on_court_memory[player_idx] = 0
            elif player_idx in on_court_memory:
                on_court_memory[player_idx] += 1

            if on_court_memory.get(player_idx, MEMORY_FRAMES + 1) > MEMORY_FRAMES:
                continue

            if player_idx not in position_history:
                position_history[player_idx]     = {LEFT_ANKLE: deque(maxlen=MOVEMENT_WINDOW),
                                                     RIGHT_ANKLE: deque(maxlen=MOVEMENT_WINDOW)}
                y_history[player_idx]            = {LEFT_ANKLE: deque(maxlen=VELOCITY_WINDOW * 3),
                                                     RIGHT_ANKLE: deque(maxlen=VELOCITY_WINDOW * 3)}
                frames_since_contact[player_idx] = {LEFT_ANKLE: 999, RIGHT_ANKLE: 999}
                player_trails[player_idx]        = {LEFT_ANKLE: [], RIGHT_ANKLE: []}

            color = PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]

            for fid in [LEFT_ANKLE, RIGHT_ANKLE]:
                x, y = int(person[fid][0]), int(person[fid][1])
                if x == 0 and y == 0:
                    continue

                position_history[player_idx][fid].append((x, y))
                y_history[player_idx][fid].append(y)
                frames_since_contact[player_idx][fid] += 1

                if not is_moving(position_history[player_idx][fid]):
                    continue

                # Trail on original frame
                trail = player_trails[player_idx][fid]
                if trail:
                    lx, ly = trail[-1]
                    if abs(x - lx) > 200 or abs(y - ly) > 200:
                        trail.clear()
                trail.append((x, y))
                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)
                for i in range(1, len(trail)):
                    thickness = max(1, int(3 * i / len(trail)))
                    cv2.line(frame, trail[i - 1], trail[i], color, thickness)

                cv2.circle(frame, (x, y), 8, color, -1)
                label = f"P{player_idx} {'L' if fid == LEFT_ANKLE else 'R'}"
                cv2.putText(frame, label, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Ground contact → warp onto flat court
                contact = detect_ground_contact(
                    y_history[player_idx][fid],
                    frames_since_contact[player_idx][fid]
                )
                if contact and is_on_court(x, y):
                    frames_since_contact[player_idx][fid] = 0
                    cv2.circle(frame, (x, y), 18, (255, 255, 255), 3)

                    wx, wy = warp_point(x, y)
                    if 0 <= wx < OUTPUT_W and 0 <= wy < OUTPUT_H:
                        pidx = min(player_idx, 1)
                        cv2.circle(heatmap[pidx], (wx, wy), 20, 1.0, -1)

            # Draw warped ankle position on live court view
            for fid in [LEFT_ANKLE, RIGHT_ANKLE]:
                x, y = int(person[fid][0]), int(person[fid][1])
                if x == 0 and y == 0 or not is_on_court(x, y):
                    continue
                wx, wy = warp_point(x, y)
                if 0 <= wx < OUTPUT_W and 0 <= wy < OUTPUT_H:
                    cv2.circle(live_court, (wx, wy), 6, color, -1)

    # ── Build combined top-down heatmap overlay ──────────────────────────────
    combined_heatmap = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)

    for pidx, hm in enumerate(heatmap):
        if hm.max() == 0:
            continue
        hm_norm  = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
        hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
        # Zero out areas with no data so they don't bleed through
        mask_hm  = (hm_norm > 0).astype(np.uint8)
        combined_heatmap = cv2.add(combined_heatmap,
                                   cv2.bitwise_and(hm_color, hm_color, mask=mask_hm))

    top_down = cv2.addWeighted(live_court, 0.6, combined_heatmap, 0.6, 0)
    draw_court_lines(top_down)   # redraw lines on top so they're always visible

    # Show original frame and top-down view side by side
    # Resize original to match top-down height for display
    scale_factor = OUTPUT_H / frame_h
    disp_w = int(frame_w * scale_factor)
    frame_resized = cv2.resize(frame, (disp_w, OUTPUT_H))
    combined = np.hstack([frame_resized, top_down])

    WIN = "Tracking | Top-Down Heatmap (drag corner to resize)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    combined_h, combined_w = combined.shape[:2]
    cv2.resizeWindow(WIN, combined_w // 2, combined_h // 2)
    cv2.imshow(WIN, combined)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

    for pid in list(on_court_memory.keys()):
        if pid not in seen_players:
            on_court_memory[pid] += 1
            if on_court_memory[pid] > MEMORY_FRAMES:
                del on_court_memory[pid]

cap.release()


# ── Step 3: save final top-down heatmap ──────────────────────────────────────
final_court = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
draw_court_lines(final_court)

combined_final = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
for pidx, hm in enumerate(heatmap):
    if hm.max() == 0:
        continue
    hm_norm  = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
    hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
    mask_hm  = (hm_norm > 0).astype(np.uint8)
    combined_final = cv2.add(combined_final,
                             cv2.bitwise_and(hm_color, hm_color, mask=mask_hm))

final = cv2.addWeighted(final_court, 0.7, combined_final, 0.8, 0)
draw_court_lines(final)   # lines on top

# Add player legend
for pidx, label in enumerate(["Player 1 (green)", "Player 2 (orange)"]):
    color = PLAYER_COLORS[pidx % len(PLAYER_COLORS)]
    cv2.putText(final, label, (10, 30 + pidx * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite("heatmap_topdown_final.png", final)
print("Saved heatmap_topdown_final.png")

cv2.imshow("Final Top-Down Heatmap", final)
cv2.waitKey(0)
cv2.destroyAllWindows()