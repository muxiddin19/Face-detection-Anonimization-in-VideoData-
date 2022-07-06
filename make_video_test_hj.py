import numpy as np
import cv2

pathIn = "./frame/AIHUB_CCTV/Assault/11-2_cam01_assault01_place08_night_spring.mp4"

cap = cv2.VideoCapture(pathIn)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()