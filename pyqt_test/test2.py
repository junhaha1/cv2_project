import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt & OpenCV Image Zoom")
        self.setGeometry(100, 100, 800, 600)

        # 기본 이미지 변수
        self.image = None
        self.zoom_factor = 1.0

        # 메인 위젯 설정
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # 레이아웃 설정
        self.layout = QVBoxLayout(self.main_widget)

        # 이미지 표시 라벨
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 확대 축소 슬라이더
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 400)  # 1% ~ 400% 사이의 값으로 확대/축소 설정
        self.zoom_slider.setValue(100)     # 기본값 100%
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        self.layout.addWidget(self.zoom_slider)

        # 이미지 불러오기 버튼
        self.open_button = QPushButton("이미지 열기")
        self.open_button.clicked.connect(self.open_image)
        self.layout.addWidget(self.open_button)

    def open_image(self):
        # 파일 열기 대화상자
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "Image Files (*.png *.jpg *.bmp)")

        if file_name:
            # OpenCV로 이미지 읽기
            self.image = cv2.imread(file_name)
            self.zoom_factor = 1.0
            self.zoom_slider.setValue(100)
            self.show_image()

    def show_image(self):
        if self.image is not None:
            # 확대/축소된 이미지 크기 계산
            height, width = self.image.shape[:2]
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            
            # 이미지를 OpenCV로 리사이즈
            resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # 이미지를 Qt 형식으로 변환
            image_qt = self.convert_cv_qt(resized_image)
            
            # 라벨에 이미지 표시
            self.image_label.setPixmap(image_qt)

    def convert_cv_qt(self, image):
        """OpenCV 이미지를 QImage로 변환하는 함수"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image.rgbSwapped())

    def update_zoom(self):
        # 슬라이더 값을 통해 확대/축소 비율 업데이트
        self.zoom_factor = self.zoom_slider.value() / 100.0
        self.show_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())
