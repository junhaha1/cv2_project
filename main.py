import os
import cv2
import numpy as np
os.chdir("C:/Users/junha/venvs/vsopencv/SourceCode/Project") #경로 수정

image = cv2.imread("image_face/45.jpg", cv2.IMREAD_COLOR)

cv2.imshow("test", image)
cv2.waitKey()