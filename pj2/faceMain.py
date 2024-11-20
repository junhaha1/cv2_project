import cv2
import numpy as np


#모션을 캡쳐
    #사용 로직
    #엄지 검지 중심 위치까지 직선 그리기
    #직선이 일정 수준보다 길어질 때 확대 처리
    #직선이 일정 수준보다 짧아질 때 축소 처리
def process_motion(frame, distance, fingers):
    pass

def calc_dist(fingers):
    f1 = fingers[0] #엄지
    f2 = fingers[1] #검지

    dx = (f1[1] - f1[0]) ** 2
    dy = (f2[1] - f2[0]) ** 2
    distance = int((dx + dy) ** 0.5)
    return distance

#화면에 초록색 영역을 추적
def tracking_green(frame, fingers):
    lower_green = np.array([40, 50, 50])  # 초록색 범위의 하한값
    upper_green = np.array([80, 255, 255])  # 초록색 범위의 상한값

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 초록색 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)
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
        if radius > 10:
            # 원의 중심 좌표 출력
            center = (int(x), int(y))
            fingers.append(center)
            #print(f"원 중심 좌표: {center}")

            # 원을 화면에 그리기
            cv2.circle(frame, center,  int(radius), (0, 0, 255), 2)

    return frame, fingers


def onMouse(event, x, y, flags, param):
    #모드 선택
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(x, y)

capture = cv2.VideoCapture(0)								# 0번 카메라 연결
if capture.isOpened() is None: raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)      # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)     # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

title = "Main Camera"              # 윈도우 이름 지정
cv2.namedWindow(title)                          # 윈도우 생성 - 반드시 생성 해야함
cv2.setMouseCallback(title, onMouse)

fingers = [] #0: 엄지, 1: 검지
distance = -1
while True:
    ret, frame = capture.read()                 # 카메라 영상 받기
    if not ret: break
    if cv2.waitKey(30) >= 0: break

    frame = cv2.flip(frame, 1) #좌우반전

    if len(fingers) == 2:
        distance = calc_dist(fingers)
    frame, fingers = tracking_green(frame.copy(), fingers)
    print(fingers)
    cv2.imshow(title, cv2.flip(frame, 1))

capture.release()