import cv2
import os

video = r"C:\Users\victo\OneDrive\Documents\Cambridge\Computing\Badminton tracker\tracking\differentMatch.mp4"

print("File exists:", os.path.exists(video))
print("File size:", os.path.getsize(video) if os.path.exists(video) else "N/A")

cap = cv2.VideoCapture(video)
print("Cap opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
print("Read success:", ret)
print("Frame:", frame if frame is None else frame.shape)
cap.release()