import cv2
import numpy as np

# Try to open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera is working! Press 'q' to close the window.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Just to show it's "thinking," let's turn it gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('VPS Camera Test', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()