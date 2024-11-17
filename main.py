import os
import cv2
import numpy as np

os.chdir("C:/Users/junha/venvs/vsopencv/SourceCode/Project") #경로 수정

def onMouse(event, x, y, flags, param):
    global line_coordinate
    if event == cv2.EVENT_LBUTTONDOWN: #좌표 선택
        if len(line_coordinate) < 4: #좌표가 4개 미만일 경우에만
            line_coordinate.append((x, y)) 
            print(len(line_coordinate))
    elif event == cv2.EVENT_RBUTTONDOWN: #좌표 선택 취소
        line_coordinate.pop()

#차선 찾기
def find_line(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    roi = frame[height//2:height, 0:width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_canny = cv2.Canny(blur, 50, 150)

    '''
    mask = np.zeros_like(gray_canny)
    vertices = np.array([[
        (width * 0.1, height),  # 좌측 하단
        (width * 0.9, height),  # 우측 하단
        (width * 0.5, height * 0.6)  # 중앙 위쪽
    ]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(gray_canny, mask)
    '''
    return draw_line(frame, roi, gray_canny)

#좌표 위치 그리기
def draw_coord(frame, line_coordinate):
    for x, y in line_coordinate:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 2, cv2.FILLED)
    return frame

#차선 그리기
def draw_line(frame, roi, edge):
    lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 녹색 선 그리기
    return frame

title = "RoadMap"
#각 레이아웃 영역 변수

#차선 감지 영역 좌표
line_coordinate = []

main_width = 1000
main_height = 600

func_width = 100
func_height = main_height

channel_width = 300
channel_height = main_height

road_width = 600
road_height = main_height

blinker_width = 300
blinker_height = 300

_mainboard = np.zeros((main_height, main_width, 3), np.uint8)
_funcboard = np.full((func_height, func_width, 3), 255, np.uint8)
_channelboard = np.full((channel_height, channel_width, 3), 255, np.uint8)

#각 영역 위치 레이아웃
_mainboard[0:600, 0:100] = _funcboard
_mainboard[0:600, 700:1000] = _channelboard

#카메라 연결 및 초기 설정 처리
road_video_path = "Videos/roadA.mp4"  
blinker_video_path = "Videos/blinker.mp4"
# VideoCapture 객체 생성
road_capture = cv2.VideoCapture(road_video_path)
blinker_capture = cv2.VideoCapture(blinker_video_path)

#capture = cv2.VideoCapture(0)								# 0번 카메라 연결
if road_capture.isOpened() is None: raise Exception("카메라 연결 안됨")

road_capture.set(cv2.CAP_PROP_FRAME_WIDTH, road_width)      # 카메라 프레임 너비
road_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, road_height)     # 카메라 프레임 높이
road_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
road_capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

blinker_capture.set(cv2.CAP_PROP_FRAME_WIDTH, blinker_width)      # 카메라 프레임 너비
blinker_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, blinker_height)     # 카메라 프레임 높이
blinker_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
blinker_capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화


while True:
    road_ret, road_frame = road_capture.read()                 # 카메라 영상 받기
    #blinker_ret, blinker_frame = blinker_capture.read()                 # 카메라 영상 받기
    if not road_ret: break

    key = cv2.waitKeyEx(30)
    if key == 27: break #esc일 경우 종료
    
    #노트북 내장 카메라가 너비와 높이 설정이 적용 안돼서 받아온 frame을 직접 사이즈 조정
    if road_frame.shape[0] != road_height or road_frame.shape[1] != road_width:
        road_frame = cv2.resize(road_frame, (600, 600), interpolation=cv2.INTER_CUBIC)
    
    if 0 < len(line_coordinate) <= 4:
        road_frame = draw_coord(road_frame, line_coordinate)
    
    #if blinker_frame.shape[0] != blinker_height or blinker_frame.shape[1] != blinker_width:
    #    blinker_frame = cv2.resize(blinker_frame, (300, 300), interpolation=cv2.INTER_LINEAR)
    #캐니 에지 검출 
    #edge = cv2.Canny(frame, 100, 150)
    #frame = cv2.bitwise_and(frame, frame, mask=edge)
    #frame = cv2.flip(frame, 1)  # 좌우 반전

    #road_frame = find_line(road_frame)
    #_mainboard[0:600, 100:700] = road_frame
    #_mainboard[0:300, 700:1000] = blinker_frame

    cv2.imshow(title, road_frame)
    cv2.setMouseCallback(title, onMouse)