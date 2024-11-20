import cv2
import numpy as np


def put_string(frame, text, pt, value, color=(120, 200, 90)):             # 문자열 출력 함수 - 그림자 효과
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)  # 그림자 효과
    cv2.putText(frame, text, pt, font, 0.7, (120, 200, 90), 2)  # 글자 적기

#모션을 캡쳐
    #사용 로직
    #엄지 검지 중심 위치까지 직선 그리기
    #직선이 일정 수준보다 길어질 때 확대 처리
    #직선이 일정 수준보다 짧아질 때 축소 처리

#확대
max_scale = 2.0  # 최대 확대 비율
min_scale = 0.5  # 최소 확대 비율

def calculate_scale_and_resize(zoomin_initial_distance, zoomout_initial_distance, current_distance, current_scale, min_scale, max_scale, smooth_factor, threshold):
    """
    초기 거리와 현재 거리, 확대 비율에 따라 이미지를 확대/축소하는 함수

    :param image: 원본 이미지
    :param initial_distance: 초기 거리 (보정 전)
    :param current_distance: 현재 거리
    :param current_scale: 현재 확대 비율
    :param min_scale: 최소 확대 비율
    :param max_scale: 최대 확대 비율
    :param smooth_factor: 부드러운 변화 비율
    :return: 조정된 이미지와 업데이트된 확대 비율
    """
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
        #scale_factor = max((current_distance - adjusted_zoomin_initial_distance) / 200.0, 0)
        #target_scale = min(max(min_scale + scale_factor, min_scale), max_scale)
        
        scale_factor = (current_distance - adjusted_zoomin_initial_distance) / 200.0
        # 현재 확대 비율을 목표 확대 비율로 점진적으로 따라가기
        updated_scale = min(current_scale + scale_factor * smooth_factor, max_scale)

    elif adjusted_zoomout_initial_distance > current_distance and (adjusted_zoomout_initial_distance - current_distance) > out_threshold:
        scale_factor = (current_distance - adjusted_zoomout_initial_distance) / 200.0
        # 현재 확대 비율을 목표 확대 비율로 점진적으로 따라가기
        updated_scale = max(current_scale + scale_factor * smooth_factor, min_scale)
    else:
        updated_scale = current_scale

    return updated_scale, in_threshold, out_threshold


def calc_dist(fingers):
    f1 = fingers[0] #엄지
    f2 = fingers[1] #검지

    dx = (f1[0] - f2[0]) ** 2
    dy = (f1[1] - f2[1]) ** 2
    distance = int((dx + dy) ** 0.5)
    return distance

#화면에 초록색 영역을 추적
def tracking_green(frame, fingers):
    lower_green = np.array([35, 30, 30])  # 초록색 범위의 하한값
    upper_green = np.array([90, 255, 255])  # 초록색 범위의 상한값

    image = cv2.GaussianBlur(frame.copy(), (5,5), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 초록색 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)

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

        # 반지름이 일정 크기 이상일 때만 처리
        if radius > 7:
            # 원의 중심 좌표 출력
            center = (int(x), int(y))
            fingers.append(center)
            #print(f"원 중심 좌표: {center}")

            # 원을 화면에 그리기
            cv2.circle(frame, center,  int(radius), (0, 0, 255), 2)

    return frame, fingers


capture = cv2.VideoCapture(0)								# 0번 카메라 연결
if capture.isOpened() is None: raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)      # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)     # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

title = "Main Camera"              # 윈도우 이름 지정
cv2.namedWindow(title)                          # 윈도우 생성 - 반드시 생성 해야함

fingers = [] #0: 엄지, 1: 검지
distance = 0
zoomin_initial_distance = 150
zoomout_initial_distance = 100

current_scale = 1.0  # 현재 확대 비율 (초기값 1.0)
target_scale = 1.0  # 목표 확대 비율
smooth_factor = 0.1  # 부드러운 변화 비율 (0.0~1.0, 낮을수록 느리게 반응)

_in = None
_out = None
while True:
    ret, frame = capture.read()                 # 카메라 영상 받기
    if not ret: break
    key = cv2.waitKey(30)
    if key == ord('q') or key == 27 : break

    frame = cv2.flip(frame, 1) #좌우반전

    if current_scale > 1:
        frame = cv2.resize(frame, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_CUBIC)
    elif current_scale < 1:
        frame = cv2.resize(frame, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_LINEAR)

    frame, fingers = tracking_green(frame.copy(), fingers)
    if len(fingers) >= 2:

        distance = calc_dist(fingers)
        #초기 거리 설정
        cv2.line(frame, fingers[0], fingers[1], (0, 0, 255), 2)

        current_scale, _in, _out = calculate_scale_and_resize(zoomin_initial_distance, zoomout_initial_distance, distance, current_scale, 0.5, 2.0, 0.07, 50)
        
        # 이미지 확대/축소
    elif len(fingers) < 2:
        fingers.clear()
        distance = 0

    put_string(frame, "distance : " , (10, 30), distance)   # 줌 값 표시
    put_string(frame, "limit zoom In : " , (10, 50), zoomin_initial_distance * current_scale)   # 줌 값 표시
    put_string(frame, "limit zoom Out : " , (10, 70), zoomout_initial_distance * current_scale)   # 줌 값 표시
    put_string(frame, "current_scale : " , (10, 90), current_scale)
    put_string(frame, "_in : " , (10, 110), _in)
    put_string(frame, "_out : " , (10, 130), _out)

    cv2.imshow(title, frame)

capture.release()