import os
import cv2

"""
Variable explanation
pathIn : cctv 영상 입력 경로
pathOut : 검출한 데이터 output 경로

"""
pathIn = "./frame/AIHUB_CCTV/Assault/11-2_cam01_assault01_place08_night_spring.mp4"
pathOut = "./Final_Data/01/"


def extractFrame(pathIn, pathOut):
    count = 0
    cap = cv2.VideoCapture(pathIn)
    success, image = cap.read()
    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        success, image = cap.read()
        print(count, "이미지 불러옴 - ", success)
        count = count + 1
        cv2.imwrite(pathOut + "img%d.jpg" % count, image)


extractFrame(pathIn, pathOut)
