import cv2
import numpy as np
import os
from os.path import isfile, join

# 유동적, 변경 필수
pathIn = "./converted/01/"
pathOut = "./Final_proc/01.mp4"
# fps = 0.5
# size 3840*2160
# fourcc
# cap.set(3,840)
# cap.set(2,160)


def frames_to_video(inputpath, outputpath, fps):
   image_array = []
   files = [f for f in os.listdir(inputpath) if isfile(join(inputpath, f))]
   count = 0
   for i in range(len(files)):
       count += 1
       print(inputpath+"proc%d"%count)
       img = cv2.imread(inputpath+"proc%d.jpg"%count)
       size =  (img.shape[1],img.shape[0])
       img = cv2.resize(img,size)
       image_array.append(img)
   fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   out = cv2.VideoWriter(outputpath,fourcc, fps, size)
   for i in range(len(image_array)):
       out.write(image_array[i])
   out.release()

fps = 29
frames_to_video(pathIn, pathOut,fps)