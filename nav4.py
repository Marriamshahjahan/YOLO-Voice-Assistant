import cv2
import torch
import pyttsx3
import threading
import queue

# Load YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Text-to-speech setup
engine = pyttsx3.init()
speech_queue = queue.Queue()

def speak_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Stop signal
            break
        engine.say(text)
        engine.runAndWait()

# Start speech thread
speech_thread = threading.Thread(target=speak_worker, daemon=True)
speech_thread.start()

def say_async(text):
    if speech_queue.empty():  # Prevent repeating too fast
        speech_queue.put(text)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    h, w, _ = frame.shape

    # Define navigation zones
    left_zone = (0, int(w/3))
    center_zone = (int(w/3), int(2*w/3))
    right_zone = (int(2*w/3), w)

    left_obstacle = False
    center_obstacle = False
    right_obstacle = False

    # Draw detections
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        obj_name = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, obj_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Center of object
        obj_center = (x1 + x2) // 2

        if left_zone[0] <= obj_center < left_zone[1]:
            left_obstacle = True
        elif center_zone[0] <= obj_center < center_zone[1]:
            center_obstacle = True
        elif right_zone[0] <= obj_center < right_zone[1]:
            right_obstacle = True

    # Draw navigation boxes
    zones = [(left_zone, "LEFT", left_obstacle),
             (center_zone, "CENTER", center_obstacle),
             (right_zone, "RIGHT", right_obstacle)]

    instruction = ""

    for zone, label, obstacle in zones:
        color = (0,255,0) if not obstacle else (0,0,255)
        cv2.rectangle(frame, (zone[0], h-100), (zone[1], h), color, -1)
        cv2.putText(frame, label, (zone[0]+40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Navigation logic with speech
    if center_obstacle:
        if not left_obstacle and right_obstacle:   # Center + Right blocked → Turn Left
            instruction = "Turn left"
        elif not right_obstacle and left_obstacle: # Center + Left blocked → Turn Right
            instruction = "Turn right"
        elif left_obstacle and right_obstacle:     # All sides blocked → Stop
            instruction = "Stop"
        else:                                      # Only center blocked → Choose Left
            instruction = "Turn left"
    else:
        if left_obstacle and not right_obstacle:   # Only left blocked → Move Right
            instruction = "Turn right"
        elif right_obstacle and not left_obstacle: # Only right blocked → Move Left
            instruction = "Turn left"
        elif left_obstacle and right_obstacle:     # Left + Right blocked but center free
            instruction = "Move forward"
        else:
            instruction = "Move forward"


    if instruction:
        cv2.putText(frame, instruction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        say_async(instruction)

    cv2.imshow("Navigation Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean exit
speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()
