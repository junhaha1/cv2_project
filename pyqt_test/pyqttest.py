from PyQt5 import QtWidgets
from PyQt5 import QtGui
import os
import cv2
import numpy as np

os.chdir("C:/Users/junha/venvs/vsopencv/SourceCode/Project") #경로 수정

image = cv2.imread("image_face/45.jpg", cv2.IMREAD_COLOR)
h,w,c = image.shape

qImg=QtGui.QImage(image.data,w,h,w*c,QtGui.QImage.Format_BGR888)

app = QtWidgets.QApplication([])

label = QtWidgets.QLabel()
pixmap = QtGui.QPixmap.fromImage(qImg)
label.setPixmap(pixmap)
label.resize(pixmap.width(),pixmap.height())
label.show()
app.exec_()