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
        elif (0 <= x <=100) and (0 <= y <= 100):
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
        elif event == cv2.EVENT_RBUTTONDOWN: #좌표 선택 취소
            if len(coordinate) > 0:
                coordinate.pop()

#차선 찾기 알고리즘 2
def find_line2(frame, mask2):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(frame, frame, mask = mask)

    masked_img = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask = mask2)
    return masked_img

#차선 찾기
def find_line(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_canny = cv2.Canny(blur, 50, 150)

    masked_edges = cv2.bitwise_and(gray_canny, mask)
    masked_edges = unsharp_image(masked_edges, 1.5, -0.5)

    return draw_line(frame, masked_edges)

#신호등 검출
def find_blinker(frame, mask):
    #frame과 차원 맞추기
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_blinker = cv2.bitwise_and(frame, mask)
    zoom_blinker = zoom_mask(masked_blinker)
    hsv_blinker = cv2.cvtColor(zoom_blinker, cv2.COLOR_BGR2HSV)

    #명도 평활화를 통해 명암을 고르게 분포하도록  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_blinker[:,:,2] = clahe.apply(hsv_blinker[:,:,2])

    #빨강 노랑 초록 색 마스크 만들기
    # 빨강 색 범위 (두 개의 범위 필요 - 빨강은 경계를 넘나듦)
    lower_red1 = np.array([0, 100, 100])   # 약간 어두운 빨강
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100]) # 약간 밝은 빨강
    upper_red2 = np.array([180, 255, 255])

    # 노랑 색 범위
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 초록 색 범위
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # 빨강 마스크 (두 개의 범위를 OR로 합침)
    mask_red1 = cv2.inRange(hsv_blinker, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_blinker, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 노랑 마스크
    mask_yellow = cv2.inRange(hsv_blinker, lower_yellow, upper_yellow)

    # 초록 마스크
    mask_green = cv2.inRange(hsv_blinker, lower_green, upper_green)
    
    result_red = cv2.cvtColor(cv2.bitwise_and(hsv_blinker, hsv_blinker, mask=mask_red), cv2.COLOR_HSV2BGR)
    result_yellow = cv2.cvtColor(cv2.bitwise_and(hsv_blinker, hsv_blinker, mask=mask_yellow), cv2.COLOR_HSV2BGR)
    result_green = cv2.cvtColor(cv2.bitwise_and(hsv_blinker, hsv_blinker, mask=mask_green), cv2.COLOR_HSV2BGR)

    RYG_pixels ={'red' : result_red,
                'yellow' : result_yellow,
                'green' : result_green}

    color = calc_pixel(RYG_pixels)
    return zoom_blinker, color

#색깔 픽셀 수 계산
def calc_pixel(pixels):
    result = 0
    color = None
    if pixels is not None:
        for col, pixel in pixels.items():
            pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2GRAY)
            count = cv2.countNonZero(pixel)
            if result < count:
                result = count
                color = col
    return color

#차량 검출
def find_safe_car(frame, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_safe_car = cv2.bitwise_and(frame, mask)

#사용자가 입력한 좌표대로 마스크 생성
def make_mask(frame, coordinate):
    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    location = []
    for loc in coordinate:
        location.append(loc)
    vertices = np.array([location], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    return mask

#언샤프닝 함수
def unsharp_image(frame, alpha, beta):
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    unsharpened_image = cv2.addWeighted(frame, alpha, blurred, beta, 0)
    return unsharpened_image

#마스크된 이미지 부분을 확대하기
def zoom_mask(masked_image):
    gray_mask = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# 마스크에서 가장 큰 사각형을 찾기 위한 컨투어 탐색
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어가 있는 경우
    if contours:
        # 가장 큰 컨투어의 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(contours[0])

        # 관심 영역(ROI) 추출
        roi = masked_image[y:y+h, x:x+w]
        scale_factor = 3
        resized_roi = cv2.resize(roi, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

        unsharpened_image = unsharp_image(resized_roi, 1.5, -0.5)
        
    return unsharpened_image

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
            #기울기가 0.5이상인 선만 그리기  => 가로선 제외
            if abs((y1 - y2) / (x1 - x2)) > 0.5:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 녹색 선 그리기
    return frame

#문자열 출력 함수
def put_string(frame, text, pt, value, color=(120, 200, 90)):         
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.5, (0, 0, 0), 2)  # 그림자 효과
    cv2.putText(frame, text, pt, font, 0.5, (120, 200, 90), 2)  # 글자 적기

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
road_mask = [] #차선 마스크
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
        #좌표가 2개 이상이고 스페이스바를 눌렀을 경우
        if 2 < len(coordinate) and key==32:
            road_mask.append(make_mask(road_frame, coordinate))
            mode = 0 #기본 모드로 초기화
            coordinate.clear()
            #cv2.imshow("test1", road_mask)
    elif mode == 2: #신호등 선택 모드
        if 2 < len(coordinate) and key==32:
            blinker_mask = make_mask(road_frame, coordinate)
            mode = 0 #기본 모드로 초기화 
            coordinate.clear()
    elif mode == 3: #차량 선택 모드
        if 2 < len(coordinate) and key==32:
            safe_car_mask_mask = make_mask(road_frame, coordinate)
            mode = 0 #기본 모드로 초기화
            coordinate.clear()
            find_safe_car(road_frame, safe_car_mask)
    
    #좌표가 있을 경우 좌표 화면에 그리기
    if 0 < len(coordinate):
        road_frame = draw_coord(road_frame, coordinate)
    
    #차선 마스크가 존재할 경우 차선 검출
    if len(road_mask) > 0:
        for mask in road_mask:
            cv2.imshow("test2", find_line2(road_frame, mask))
            road_frame = find_line(road_frame, mask)

    #신호등 마스크가 존재할 경우 신호등 색상 문구 출력
    if blinker_mask is not None:
        blinker_frame, color = find_blinker(road_frame, blinker_mask)
        _mainboard[0:150, 700:850] = cv2.resize(blinker_frame, (150, 150), interpolation=cv2.INTER_LINEAR)
        if color is None:
            put_string(road_frame, "Current Blinker : " , (10, 80), " ")   
        else: 
            put_string(road_frame, "Current Blinker : " , (10, 80), color)   
    
    #캐니 에지 검출 
    #edge = cv2.Canny(frame, 100, 150)
    #frame = cv2.bitwise_and(frame, frame, mask=edge)
    #frame = cv2.flip(frame, 1)  # 좌우 반전

    put_string(road_frame, "Current Mode : " , (10, 50), mode_text[mode])   
    _mainboard[0:600, 100:700] = road_frame

    cv2.imshow(title, _mainboard)
    cv2.setMouseCallback(title, onMouse)