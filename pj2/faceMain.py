import cv2
import numpy as np

def put_string(frame, text, pt, value="", color=(120, 200, 90), size=0.7):             # 문자열 출력 함수 - 그림자 효과
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)  # 그림자 효과
    cv2.putText(frame, text, pt, font, size, color, 2)  # 글자 적기

def calculate_scale_and_resize(zoomin_initial_distance, zoomout_initial_distance, current_distance, current_scale, min_scale, max_scale, smooth_factor, threshold):
    # 초기 거리 보정 (확대 비율로 원래 거리 복원)
    adjusted_zoomin_initial_distance = zoomin_initial_distance * current_scale
    adjusted_zoomout_initial_distance = zoomout_initial_distance * current_scale

    if current_scale > 1:
        in_threshold = threshold * current_scale
        out_threshold = threshold / (current_scale * 10)
    else:
        in_threshold = threshold / current_scale
        out_threshold = threshold * (current_scale / 10)

    if current_distance > adjusted_zoomin_initial_distance and (current_distance - adjusted_zoomin_initial_distance) > in_threshold:
        scale_factor = (current_distance - adjusted_zoomin_initial_distance) / 200.0
        # 현재 확대 비율을 목표 확대 비율로 점진적으로 따라가기
        updated_scale = min(current_scale + scale_factor * smooth_factor, max_scale)

    elif adjusted_zoomout_initial_distance > current_distance and (adjusted_zoomout_initial_distance - current_distance) > out_threshold:
        scale_factor = (current_distance - adjusted_zoomout_initial_distance) / 200.0
        # 현재 확대 비율을 목표 확대 비율로 점진적으로 따라가기
        updated_scale = max(current_scale + scale_factor * smooth_factor, min_scale)
    else:
        updated_scale = current_scale

    return updated_scale

#이동 시에 화면 이동 좌표 보정
def correct_move_location(move_x, move_y, dx, dy, frame_width, frame_height, threshold):
    if abs(dx) > threshold:
        move_x = max(0, min(move_x - dx, frame.shape[1] - frame_width))
    if abs(dy) > threshold:
        move_y = max(0, min(move_y - dy, frame.shape[0] - frame_height))

    return int(move_x), int(move_y)

#확대, 축소시에 좌표 보정
def correct_location(fingers, current_scale, initial_width, initial_height):
    center_x, center_y = calc_center(fingers)
    new_center_x, new_center_y = int(center_x * current_scale), int(center_y * current_scale) 
    new_width, new_height = int(initial_width * current_scale), int(initial_height * current_scale)

    move_x = min((new_center_x - center_x), (new_width - initial_width))
    move_y = min((new_center_y - center_y), (new_height - initial_height))

    if move_x < 0: move_x = 0
    if move_y < 0: move_y = 0

    return (move_x, move_y)

#계산 관련 함수
#엄지와 검지의 거리 계산
def calc_dist(fingers): 
    f1 = fingers[0] #엄지
    f2 = fingers[1] #검지

    dx = (f1[0] - f2[0]) ** 2
    dy = (f1[1] - f2[1]) ** 2
    distance = int((dx + dy) ** 0.5)
    return distance

#엄지와 검지의 중점의 좌표 계산
def calc_center(fingers):
    f1 = fingers[0] #엄지
    f2 = fingers[1] #검지

    center_x = (f1[0] + f2[0]) // 2
    center_y = (f1[1] + f2[1]) // 2

    return (center_x, center_y)

#마스크 제작
def tracking_mask(mask, center_x, center_y, radius=10):
    
    mask = cv2.circle(mask, (center_x, center_y),  radius, 255, cv2.FILLED)
    return mask

#화면에 초록색 영역을 추적
def tracking_color(frame, fingers, lower, upper, initial_radius=None):

    image = cv2.GaussianBlur(frame.copy(), (5,5), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #해당 색 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv, lower, upper)

    #닫힘 연산을 통해 노이즈를 제거하기
    open_mask = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]]).astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_mask) 

    # 마스크에 대해 윤곽선을 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    fingers.clear() #리스트 초기화
    for contour in contours:
        # 윤곽선의 영역이 너무 작으면 무시
        if cv2.contourArea(contour) < 500:
            continue

        # 외곽에 외접하는 원 찾기
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        #반지름이 일정 길이 이상일 때
        if radius > 10:
            # 원의 중심 좌표 출력
            center = (int(x), int(y))
            fingers.append(center)

            if initial_radius is None: #초기값이 없을 경우 감지된 반지름으로
                cv2.circle(frame, center,  int(radius), (0, 0, 255), 2)
            else:
                cv2.circle(frame, center,  initial_radius, (0, 0, 255), 2)

    return frame, fingers

capture = cv2.VideoCapture(0)								# 0번 카메라 연결
if capture.isOpened() is None: raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)      # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)     # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

title = "Main Camera"              # 윈도우 이름 지정
cv2.namedWindow(title)                          # 윈도우 생성 - 반드시 생성 해야함

#확대
max_scale = 2.0  # 최대 확대 비율
min_scale = 0.5  # 최소 확대 비율

mode_name = ["common", "zoom", "move", "blur"]
mode = 0

fingers = [] #0: 엄지, 1: 검지
distance = 0

zoomin_initial_distance = 150
zoomout_initial_distance = 100

main_width = 1000
main_height = 500

frame_width = 640
frame_height = 360

current_scale = 1.0  # 현재 확대 비율 (초기값 1.0)

move_x, move_y = 0,0

ZOOM_THRESHOLD = 50 #확대 축소 시 임계치
MOVE_THRESHOLD = 10 #화면 이동 시 임계치

current_finger_position = None
previous_finger_position = None

blured_mask = None
blured_frame = None
blured_size = 10

lower_green = np.array([35, 50, 50])  # 초록색 범위의 하한값
upper_green = np.array([90, 255, 255])  # 초록색 범위의 상한값

_mainboard = np.zeros((main_height, main_width, 3), np.uint8)

while True:
    ret, frame = capture.read()                 # 카메라 영상 받기
    if not ret: break

    #키보드를 통한 모드 설정
    key = cv2.waitKey(30)
    if key == ord('q') or key == 27: 
        break
    elif key == ord('r'):
        move_x, move_y = 0,0
        current_scale = 1.0
        mode = 0
        blured_mask = None
        blured_frame = None
        blured_size = 10
    elif key == ord('c'):
        fingers.clear()
        previous_finger_position = None
        distance = 0
        mode = 0
    elif key == ord('z'):
        fingers.clear()
        distance = 0
        mode = 1
    elif key == ord('m'):
        fingers.clear()
        distance = 0
        mode = 2
    elif key == ord('b'):
        fingers.clear()
        distance = 0
        mode = 3
    elif key == ord('u'):
        blured_size = min(blured_size + 1, 100)
    elif key == ord('d'):
        blured_size = max(blured_size - 1, 1)
    
    #초반 기본 화면들 설정
    frame = cv2.flip(frame, 1) #좌우반전
    _mainboard.fill(255)

    if current_scale > 1:
        frame = cv2.resize(frame, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_CUBIC)
    elif current_scale < 1:
        frame = cv2.resize(frame, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_LINEAR)

    #확대&축소 모드
    if mode == 1:
        frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green)

        if len(fingers) >= 2:

            distance = calc_dist(fingers)
            #초기 거리 설정
            cv2.line(frame, fingers[0], fingers[1], (0, 0, 255), 2)
            current_scale = calculate_scale_and_resize(zoomin_initial_distance, zoomout_initial_distance, distance, current_scale, 0.5, 2.0, 0.07, ZOOM_THRESHOLD)

            if current_scale != 1.0:
                #좌표 보정하기
                move_x, move_y= correct_location(fingers, current_scale, frame_width, frame_height)

            # 이미지 확대/축소
        elif len(fingers) < 2:
            fingers.clear()
            distance = 0
    #화면 이동 모드 => 화면이 확대되었을 때에만 작동
    elif mode == 2 and current_scale > 1.0:
        frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green, initial_radius=15)
        if len(fingers) == 1:
            current_finger_position = fingers[0]
            
            if previous_finger_position is not None:
                dx = current_finger_position[0] - previous_finger_position[0]
                dy = current_finger_position[1] - previous_finger_position[1]
                move_x, move_y = correct_move_location(move_x, move_y, dx, dy, frame_width, frame_height, MOVE_THRESHOLD)

            # 현재 검지 좌표와 시간을 이전 값으로 갱신
            previous_finger_position = current_finger_position
        else:
            previous_finger_position = None
    #블러링 => 블러링을 적용할 마스크만 생성
    elif mode == 3: 
        frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green, initial_radius = blured_size)
        put_string(_mainboard, "Blur Size : ", (_mainboard.shape[1] // 2, 55), blured_size, color=(0, 0, 0), size=0.6)
        put_string(_mainboard, "'u' : Blur Size UP ", (_mainboard.shape[1] // 2, 15), color=(255, 0, 0), size=0.6)
        put_string(_mainboard, "'d' : Blur Size DOWN ", (_mainboard.shape[1] // 2, 35), color=(0, 0, 255), size=0.6)
        if blured_mask is None: #마스크가 없을 경우 초기화
            blured_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        if len(fingers) == 1:
            center = fingers[0]
            blured_mask = tracking_mask(blured_mask, center[0], center[1], blured_size)
    
    #마스크가 있을 경우 해당 마스크를 계속 적용시켜주기
    if blured_mask is not None:
        blured_mask = cv2.resize(blured_mask, (frame.shape[1], frame.shape[0]))
        target_frame = cv2.bitwise_and(frame.copy(), frame.copy(), mask=blured_mask)
        blured_frame = cv2.GaussianBlur(target_frame, (15, 15), 0)
        frame[blured_mask > 0] = blured_frame[blured_mask > 0]

    move_frame = frame[move_y:move_y + frame_height, move_x:move_x + frame_width]

    # _mainboard 중앙에 move_frame을 배치하기 위한 계산
    _mainboard_center_x = _mainboard.shape[1] // 2
    _mainboard_center_y = _mainboard.shape[0] // 2
    frame_center_x = move_frame.shape[1] // 2
    frame_center_y = move_frame.shape[0] // 2

    # _mainboard 중앙에 frame을 배치하기 위한 시작 좌표 계산
    start_x = _mainboard_center_x - frame_center_x
    start_y = _mainboard_center_y - frame_center_y

    # 시작 좌표가 음수일 경우, 최소값을 0으로 설정
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0

    # _mainboard 중앙에 frame 배치
    end_y = min(start_y + move_frame.shape[0], _mainboard.shape[0])
    end_x = min(start_x + move_frame.shape[1], _mainboard.shape[1])

    put_string(_mainboard, "distance : ", (180, 15), distance, color=(0,0,0), size=0.6) 
    put_string(_mainboard, "mode : ", (180, 35), mode_name[mode], color=(255, 0,0), size=0.6)
    put_string(_mainboard, "current_scale : ", (180, 55), round(current_scale, 2), color=(0, 0, 255), size=0.6)
    

    #put_string(_mainboard, "limit zoom In : ", (10, 50), round(zoomin_initial_distance * current_scale, 2))  # 줌 기준 값 표시
    #put_string(_mainboard, "limit zoom Out : ", (10, 70), round(zoomout_initial_distance * current_scale, 2))  # 축소 기준 값 표시

    put_string(_mainboard, "<Keyboard>", (10, 90), color=(0,0,0))
    put_string(_mainboard, "'c' : Common", (10, 120), color=(0,0,0))
    put_string(_mainboard, "'r' : Reset", (10, 150), color=(0,0,0))
    put_string(_mainboard, "'z' : Zoom", (10, 180), color=(0,0,0))
    put_string(_mainboard, "'m' : Move", (10, 210), color=(0,0,0))
    put_string(_mainboard, "'b' : Blur", (10, 230), color=(0,0,0))
    
    _mainboard[start_y:end_y, start_x:end_x] = move_frame

    cv2.imshow(title, _mainboard)

capture.release()