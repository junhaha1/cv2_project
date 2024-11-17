import os
import cv2
import numpy as np

os.chdir("C:/Users/junha/venvs/vsopencv/SourceCode/Project") #경로 수정

def onMouse(event, x, y, flags, param):
    global coordinate, mode

    #모드 선택
    if event == cv2.EVENT_LBUTTONDOWN: 
        if (0 <= x <=100) and (0 <= y <= 100):
            mode = 1 #차선 좌표 선택 모드
            coordinate.clear()
        elif (0 <= x <=100) and (100 <= y <= 200):
            mode = 2 #신호등 좌표 선택 모드
            coordinate.clear()
        elif (0 <= x <=100) and (100 <= y <= 300):
            mode = 3 #차량 좌표 선택 모드
            coordinate.clear()

    #기본 모드로 초기화
    if event == cv2.EVENT_MOUSEWHEEL:
        mode = 0 #차선 좌표 선택 모드
        coordinate.clear()

    #기본 모드가 아닐 경우 좌표 수집
    if mode != 0:
        if event == cv2.EVENT_LBUTTONDOWN: #좌표 선택
            if 100 <= x <= 700:
                coordinate.append((x - 100, y)) 
                print(len(coordinate))
        elif event == cv2.EVENT_RBUTTONDOWN: #좌표 선택 취소
            if len(coordinate) > 0:
                coordinate.pop()

#차선 찾기
def find_line(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_canny = cv2.Canny(blur, 50, 150)

    masked_edges = cv2.bitwise_and(gray_canny, mask)
    return draw_line(frame, masked_edges)

#사용자가 입력한 좌표대로 마스크 생성
def make_mask(frame, coordinate):
    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    location = []
    for loc in coordinate:
        location.append(loc)
    vertices = np.array([location], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    return mask

#레이아웃에 아이콘 그리기
def draw_icons(frame):
    file_name = ('road_icon.jpg',
                'blinker_icon.jpg',
                'car_icon.jpg'
                )
    y = 0
    x = 100
    for file in file_name:
        icon = cv2.imread('Images/' + file, cv2.IMREAD_COLOR)
        icon = cv2.resize(icon, (100, 100), interpolation=cv2.INTER_LINEAR)
        frame[y:y + 100, 0:x] = icon
        y += 100
    return frame

#좌표 위치 그리기
def draw_coord(frame, line_coordinate):
    for x, y in line_coordinate:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 2, cv2.FILLED)
    return frame

#차선 그리기
def draw_line(frame, edge):
    lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 녹색 선 그리기
    return frame

#문자열 출력 함수
def put_string(frame, text, pt, value, color=(120, 200, 90)):         
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)  # 그림자 효과
    cv2.putText(frame, text, pt, font, 0.7, (120, 200, 90), 2)  # 글자 적기

#텍스트
title = "RoadMap"
mode_text = ("Common Mode",
            "Line Select Mode",
            "Blinker Select Mode",
            "Car Select Mode"
            )

#좌표 영역
coordinate = []
mode = 0 #모드

#마스크
road_mask = None #차선 마스크
blinker_mask = None #신호등 마스크
safe_car_mask = None #안전거리 차량 마스크

#각 레이아웃 영역 변수
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

#각 영역 보드
_mainboard = np.zeros((main_height, main_width, 3), np.uint8)
_funcboard = np.full((func_height, func_width, 3), 255, np.uint8)
_channelboard = np.full((channel_height, channel_width, 3), 255, np.uint8)

#각 영역 위치 레이아웃
_funcboard = draw_icons(_funcboard)
_mainboard[0:600, 0:100] = _funcboard
_mainboard[0:600, 700:1000] = _channelboard

#카메라 연결 및 초기 설정 처리
road_video_path = "Videos/v2.mp4"  

# VideoCapture 객체 생성
road_capture = cv2.VideoCapture(road_video_path)

#capture = cv2.VideoCapture(0)								# 0번 카메라 연결
if road_capture.isOpened() is None: raise Exception("카메라 연결 안됨")

road_capture.set(cv2.CAP_PROP_FRAME_WIDTH, road_width)      # 카메라 프레임 너비
road_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, road_height)     # 카메라 프레임 높이
road_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
road_capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

while True:
    road_ret, road_frame = road_capture.read()                 # 카메라 영상 받기
    #blinker_ret, blinker_frame = blinker_capture.read()                 # 카메라 영상 받기
    if not road_ret: break

    key = cv2.waitKeyEx(30)
    if key == 27: break #esc일 경우 종료
    
    #노트북 내장 카메라가 너비와 높이 설정이 적용 안돼서 받아온 frame을 직접 사이즈 조정
    if road_frame.shape[0] != road_height or road_frame.shape[1] != road_width:
        road_frame = cv2.resize(road_frame, (600, 600), interpolation=cv2.INTER_CUBIC)

    #메인 프레임 모드별 활성화
    if mode == 0: #기본 모드
        pass
    elif mode == 1: #차선 선택 모드
        pass
    elif mode == 2: #신호등 선택 모드
        pass 
    elif mode == 3: #차량 선택 모드
        pass
    
    #좌표가 있을 경우 좌표 화면에 그리기
    if 0 < len(coordinate):
        road_frame = draw_coord(road_frame, coordinate)
    
    #좌표가 2개 이상이고 스페이스바를 눌렀을 경우
    if 2 < len(coordinate) and key==32:
        road_mask = make_mask(road_frame, coordinate)
        mode = 0 #기본 모드로 초기화
        cv2.imshow("test", road_mask)

    #차선 마스크가 존재할 경우 차선 검출
    if road_mask is not None:
        road_frame = find_line(road_frame, road_mask)
    
    #캐니 에지 검출 
    #edge = cv2.Canny(frame, 100, 150)
    #frame = cv2.bitwise_and(frame, frame, mask=edge)
    #frame = cv2.flip(frame, 1)  # 좌우 반전

    put_string(road_frame, "Current Mode : " , (10, 50), mode_text[mode])   # 줌 값 표시
    _mainboard[0:600, 100:700] = road_frame

    cv2.imshow(title, _mainboard)
    cv2.setMouseCallback(title, onMouse)