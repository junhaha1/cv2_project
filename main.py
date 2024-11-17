import os
import cv2
import numpy as np

os.chdir("C:/Users/junha/venvs/vsopencv/SourceCode/Project") #경로 수정

#차선 찾기
def find_line(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    gray_roi = frame[height//2:height-110, 0:width]
    gray = cv2.cvtColor(gray_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_canny = cv2.Canny(blur, 100, 150)

    return gray_canny

#각 레이아웃 영역 변수
main_width = 1000
main_height = 600

func_width = 100
func_height = main_height

channel_width = 300
channel_height = main_height

frame_width = 600
frame_height = main_height

_mainboard = np.zeros((main_height, main_width, 3), np.uint8)
_funcboard = np.full((func_height, func_width, 3), 255, np.uint8)
_channelboard = np.full((channel_height, channel_width, 3), 255, np.uint8)

#각 영역 위치 레이아웃
_mainboard[0:600, 0:100] = _funcboard
_mainboard[0:600, 700:1000] = _channelboard

#카메라 연결 및 초기 설정 처리
video_path = "Videos/v2.mp4"  # 동영상 파일 경로를 지정하세요.
# VideoCapture 객체 생성
capture = cv2.VideoCapture(video_path)

#capture = cv2.VideoCapture(0)								# 0번 카메라 연결
if capture.isOpened() is None: raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)      # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)     # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

while True:
    ret, frame = capture.read()                 # 카메라 영상 받기
    if not ret: break
    if cv2.waitKey(30) >= 0: break
    
    #노트북 내장 카메라가 너비와 높이 설정이 적용 안돼서 받아온 frame을 직접 사이즈 조정
    if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
        frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_CUBIC)
    
    #캐니 에지 검출 
    #edge = cv2.Canny(frame, 100, 150)
    #frame = cv2.bitwise_and(frame, frame, mask=edge)
    #frame = cv2.flip(frame, 1)  # 좌우 반전

    frame = find_line(frame)
    #_mainboard[0:600, 100:700] = frame

    cv2.imshow("test", frame)