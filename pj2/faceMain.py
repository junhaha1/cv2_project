import cv2
import numpy as np

# 문자열 출력 함수
def put_string(frame, text, pt, value="", color=(120, 200, 90), size=0.7):
    text += str(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, pt, font, size, color, 2) 

#확대, 축소 비율 계산 함수
def calculate_scale(zoomin_initial_distance, zoomout_initial_distance, current_distance, current_scale, min_scale, max_scale, smooth_factor, threshold):
    #기준 거리 보정 => 화면에 감지되는 손가락 사이의 거리와의 같은 비율로 맞춰주기
    adjusted_zoomin_initial_distance = zoomin_initial_distance * current_scale
    adjusted_zoomout_initial_distance = zoomout_initial_distance * current_scale

    #확대된 상태에선 확대보단 축소에 더 민감하도록 보정
    #축소된 상태에선 축소보단 확대에 더 민감하도록 보정
    #현재 비율이 1보다 크다면(확대) 확대 임계치는 더 커지도록, 축소 임계치는 더 작아지도록 수행
    if current_scale > 1:
        in_threshold = threshold * current_scale
        out_threshold = threshold / (current_scale * 10)
    #현재 비율이 1보다 작다면(축소) 확대 임계치는 더 작아지도록, 축소 임계치는 더 커지도록 수행
    else:
        in_threshold = threshold / current_scale
        out_threshold = threshold * (current_scale / 10)

    #확대 시에 화면 비율 계산 수행
    if current_distance > adjusted_zoomin_initial_distance and (current_distance - adjusted_zoomin_initial_distance) > in_threshold:
        # 확대 시에 비율이 점차적으로 늘어나도록 계산
        scale_factor = (current_distance - adjusted_zoomin_initial_distance) / 200.0
        # 현재 확대 비율이 최대 확대 비율을 넘어가지 않도록 제한
        updated_scale = min(current_scale + scale_factor * smooth_factor, max_scale)

    #축소 시에 화면 비율 계산 수행
    elif adjusted_zoomout_initial_distance > current_distance and (adjusted_zoomout_initial_distance - current_distance) > out_threshold:
        # 축소 시에 비율이 점차적으로 줄어들도록 계산
        scale_factor = (current_distance - adjusted_zoomout_initial_distance) / 200.0
        # 현재 확대 비율이 최소 축소 비율보다 작아지지 않도록 제한
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

#해당 좌표에 마스크 갱신
def tracking_mask(mask, center_x, center_y, radius=10):
    
    mask = cv2.circle(mask, (center_x, center_y),  radius, 255, cv2.FILLED)
    return mask

#해당 좌표에 마스크 지우기
def eraser_mask(mask1, mask2, center_x, center_y, radius=10):
    mask1 = cv2.circle(mask1, (center_x, center_y),  radius, 0, cv2.FILLED)
    mask2 = cv2.circle(mask2, (center_x, center_y),  radius, 0, cv2.FILLED)
    return mask1, mask2

#화면에 초록색 영역을 추적
def tracking_color(frame, fingers, lower, upper, initial_radius=None, target_frame=None):

    #가우시안으로 노이즈 제거
    image = cv2.GaussianBlur(frame.copy(), (5,5), 0)
    #영상 이미지를 HSV로 변환
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

        #해당 윤곽선에 외접하는 원 찾기
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        #반지름이 일정 길이 이상일 때
        if radius > 10:
            # 원의 중심 좌표 출력
            center = (int(x), int(y))
            fingers.append(center)

            if initial_radius is None: #초기값이 없을 경우 감지된 반지름으로
                if target_frame is None:
                    cv2.circle(frame, center,  int(radius), (0, 0, 255), 2)
                else:
                    cv2.circle(target_frame, center,  int(radius), (0, 0, 255), 2)
            else:
                if target_frame is None:
                    cv2.circle(frame, center,  initial_radius, (0, 0, 255), 2)
                else:
                    cv2.circle(target_frame, center,  initial_radius, (0, 0, 255), 2)

    if target_frame is None:
        return frame, fingers
    else:
        return target_frame, fingers

#해당 영역 샤프닝 적용 (선명화)
def apply_sharpening(i, image, img_mask):
    masks = [
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        [[-1, -1, -1],
         [-1, 9, -1],
         [-1, -1, -1]],
        [[1, -2, 1],
         [-2, 5, -2],
         [1, -2, 1]]
    ]
    sharp_mask = np.array(masks[i], np.float32)
    img_mask = cv2.resize(img_mask, (image.shape[1], image.shape[0]))
    target_image = cv2.bitwise_and(image.copy(), image.copy(), mask=img_mask)
    sarped_image = cv2.filter2D(target_image, -1, sharp_mask)
    image[img_mask > 0] = sarped_image[img_mask > 0]

    return image

def apply_canny(image, img_mask): #캐니 엣지를 통한 컬러 카툰 렌더링 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    edges = cv2.Canny(gray, 100, 200)  
    edges = cv2.bitwise_not(edges)  

    img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  
    cartoon = cv2.bitwise_and(image, img_edges, mask=img_mask) 
    image[img_mask > 0] = cartoon[img_mask > 0]

    return image

def apply_perspective(image, dots): #원근감 적용
        # 좌표 4개 중 상하좌우 찾기
    sm = dots.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(dots, axis=1)  # 4쌍의 좌표 각각 x-y 계산

    topLeft = dots[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
    bottomRight = dots[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
    topRight = dots[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = dots[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

    # 변환 전 4개 좌표 
    dots1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = int(max([w1, w2]))  # 두 좌우 거리간의 최대값이 서류의 폭
    height = int(max([h1, h2]))  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    dots2 = np.float32([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(dots1, dots2)
    # 원근 변환 적용
    image = cv2.warpPerspective(image, mtrx, (width, height))

    return image

def apply_contrast(): #명암 조절 적용
    pass




##################관련 변수 및 상수######################
###기본 화면 너비, 높이 초기값###
#메인보드
main_width = 1000
main_height = 500

#영상 프레임
frame_width = 640
frame_height = 360

max_scale = 2.0  # 최대 확대 비율
min_scale = 0.5  # 최소 확대 비율
current_scale = 1.0  # 현재 확대 비율 (초기값 1.0)

#모드 관련 설정
mode_name = ["common", "zoom", "move", "blur", "sharp", "eraser", "cartoon", "perspective"]
mode = 0
previous_key = 0

#손가락 좌표 및 거리 관련 변수
fingers = [] #0: 엄지, 1: 검지
distance = 0 #두 손가락 사이 거리

#확대 축소를 판단하는 기준 거리
zoomin_initial_distance = 150
zoomout_initial_distance = 100

#화면 이동 시 이동한 좌표 값
move_x, move_y = 0,0

#화면 이동 시 사용하는 손가락 좌표
current_finger_position = None
previous_finger_position = None

ZOOM_THRESHOLD = 50 #확대 축소 시 임계치
MOVE_THRESHOLD = 10 #화면 이동 시 임계치

#블러 모드 시에 사용할 변수
blured_mask = None #블러 마스크
blured_frame = None #블러 마스크에서 적용된 블러 이미지
target_size = 10 #블러 수행 사이즈

#샤프닝 모드 시에 사용할 변수
sharped_mask = None

#캐니 엣지를 통한 카툰 렌더링 관련 변수
canny_mask = None

lower_green = np.array([35, 50, 50])  # 초록색 범위의 하한값
upper_green = np.array([90, 255, 255])  # 초록색 범위의 상한값

#캡쳐 모드 관련 변수
capture_list = [] #캡쳐한 이미지 담아두는 리스트
temp_list = [] #현재 수정 작업 중인 이미지
result_list = [] #수정을 완료한 캡쳐 리스트

#토글 관련 변수
sub_frame = None
toggle = False
side_gap = 10

#원근법 관련 변수
dots = []

#메인 보드 생성
_programboard = np.zeros((main_height, 1300, 3), np.uint8) #최종 프로그램 화면
b, g, r =cv2.split(_programboard)
g[:] = 255
_programboard = cv2.merge((b,g,r)) #최종 프로그램 결과 창

_mainboard = np.zeros((main_height, main_width, 3), np.uint8) #메인 카메라, 수정할 수 있는 화면
_resultboard = np.zeros((main_height, 300, 3), np.uint8) #결과 화면
_sideboard = np.zeros((main_height, (main_width - frame_width)//2, 3), np.uint8) #캡쳐 리스트 보여주는 화면
#####################################################

###카메라 기본 설정###
#capture = cv2.VideoCapture(0)   # 0번 기본 노트북 웹캡 카메라 연결
capture = cv2.VideoCapture(0)   # 1번 외부 카메라 연결
if not capture.isOpened():
    print("카메라가 열리지 않습니다.")
    exit()

capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)      # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)     # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 오토포커싱 중지
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

title = "Main Camera"
cv2.namedWindow(title)

###메인 루프###
while True:
    #카메라 영상 읽어오기
    ret, frame = capture.read()
    if not ret:
        print("프레임 읽어오기 실패.")
        break

    #키보드를 통한 모드 설정
    key = cv2.waitKey(30)
    #'q' or 'esc' 누를 시에 종료
    if key == ord('q') or key == 27: 
        break
    elif key == ord('r'): #모든 설정 초기화
        move_x, move_y = 0,0
        current_scale = 1.0
        mode = 0
        blured_mask = None
        blured_frame = None
        target_size = 10

        capture_list.clear()
        temp_list.clear()
        result_list.clear()

        dots.clear()
        sharped_mask = None
        canny_mask = None

    elif key == ord('o'): #모든 기본 모드 주요 변경사항만 유지
        fingers.clear()
        dots.clear()
        previous_finger_position = None
        distance = 0
        mode = 0
    elif key == ord('z'): #확대 축소 모드
        fingers.clear()
        distance = 0
        mode = 1
    elif key == ord('m'): #화면 이동 모드
        fingers.clear()
        distance = 0
        mode = 2
    elif key == ord('b'): #블러 모드
        previous_key = key
        fingers.clear()
        distance = 0
        mode = 3
    elif key == ord('s'): #샤프닝 모드
        previous_key = key
        fingers.clear()
        distance = 0
        mode = 4
    elif key == ord('e'): #지우개 모드
        previous_key = key
        fingers.clear()
        distance = 0
        mode = 5
    elif key == ord('k'): #캐니엣지(카툰렌더링) 모드
        previous_key = key
        fingers.clear()
        distance = 0
        mode = 6
    elif toggle and key == ord('p'): #원근감 보정 모드 => 토글 돼었을 시에만 사용가능
        previous_key = key
        fingers.clear()
        distance = 0
        mode = 7
    #블러 원 사이즈 조절
    elif (previous_key == ord('k') or previous_key == ord('e') or previous_key == ord('b') or previous_key == ord('s')) and key == ord('u'):
        target_size = min(target_size + 1, 100)
    elif (previous_key == ord('k') or previous_key == ord('e') or previous_key == ord('b') or previous_key == ord('s')) and key == ord('d'):
        target_size = max(target_size - 1, 1)

    elif key == ord('t'): #캡쳐한 이미지와 토글 버튼
        mode = 0 #토글 시에 초기 모드는 기본 모드
        dots.clear()
        if len(capture_list) > 0:
            toggle = not toggle
        else:
            toggle = False
    elif key == 13: #엔터 눌렀을 시에 수정된 이미지를 결과 리스트에 추가:
        if toggle and len(temp_list) > 0:
            result_list.append(temp_list.pop())
            print(len(result_list))
            toggle = False

    elif len(capture_list) > 0 and key == 45: #'-' 눌렀을 경우 캡쳐 리스트 삭제하기:
        capture_list.remove(capture_list[0])

    ###초반 메인 화면, 메인 프레임 설정###
    #사이드 보드 화면 설정 => 토글 여부로
    _mainboard.fill(255)       #메인모드 흰 색으로 초기화
    _resultboard.fill(255)     #결과보드 흰색으로 초기화
    _sideboard.fill(0)         #사이드바 검은 색으로 초기화

    #토글 및 캡쳐 리스트에 대한 초기 설정
    if toggle and len(capture_list) > 0:
        side_y = 10
        
        sub_frame = frame.copy()

        if len(temp_list) > 0:
            frame = temp_list.pop()
        else:
            frame = capture_list[-1].copy()

        temp_frame = frame.copy() #임시 리스트에 넣어둘 이미지

        #토글 했을 시에 캡쳐 리스트에 있는 이미지를 메인 프레임에
        #실시간 영상은 결과 보드 화면에 출력
        resized_sub_frame = cv2.resize(sub_frame.copy(), (300,160), interpolation=cv2.INTER_LINEAR)
        _resultboard[0:160, 0:300] = resized_sub_frame
    elif not toggle:
        side_y = 10
        sub_frame = None

    #캡처 리스트를 사이드 보드에 출력
    if len(capture_list) > 0:
        for i, img in enumerate(capture_list):
            if side_y >= _sideboard.shape[0]:
                break
            resized_img = cv2.resize(img.copy(), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
            x = (_sideboard.shape[1] - resized_img.shape[1]) // 2
            _sideboard[side_y:side_y+resized_img.shape[0],x:x + resized_img.shape[1]] = resized_img
            side_y += resized_img.shape[0] + side_gap

    #수정된 캡쳐 이미지 결과 보드에 출력
    #테스트 코드
    if len(result_list) > 0:
        img = cv2.resize(result_list[-1].copy(), (300,160), interpolation=cv2.INTER_LINEAR)
        y = _resultboard.shape[0] // 2
        _resultboard[y:y+img.shape[0],0:img.shape[1]] = img
    

    #확대 축소로 변경된 화면 비율일 때 해당 비율만큼 화면 사이즈 수정
    if current_scale > 1:
        frame = cv2.resize(frame, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_CUBIC)
    elif current_scale < 1:
        frame = cv2.resize(frame, None, fx=current_scale, fy=current_scale, interpolation=cv2.INTER_LINEAR)
    ########################

    ###각 모드 수행 조건문###
    #확대&축소 모드
    if mode == 1:
        if sub_frame is not None: #토글이 되었을 경우
            frame, fingers = tracking_color(sub_frame.copy(), fingers, lower_green, upper_green, target_frame=frame)
        else: #토글되지 않았을 경우
            frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green)
            if len(fingers) >= 2:
                distance = calc_dist(fingers)
                #초기 거리 설정
                cv2.line(frame, fingers[0], fingers[1], (0, 0, 255), 2)
                current_scale = calculate_scale(zoomin_initial_distance, zoomout_initial_distance, distance, current_scale, 0.5, 2.0, 0.07, ZOOM_THRESHOLD)

                if current_scale != 1.0:
                    #좌표 보정하기
                    move_x, move_y= correct_location(fingers, current_scale, frame_width, frame_height)

            elif len(fingers) < 2:
                fingers.clear()
                distance = 0
    #화면 이동 모드 => 화면이 확대되었을 때에만 작동
    elif mode == 2 and current_scale > 1.0:
        if sub_frame is not None:
            frame, fingers = tracking_color(sub_frame.copy(), fingers, lower_green, upper_green, initial_radius=15, target_frame=frame)
        else:
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
        if sub_frame is not None: #토글 됐을 경우
            frame, fingers = tracking_color(sub_frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size, target_frame=frame)
        else:
            frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size)
            if blured_mask is None: #마스크가 없을 경우 초기화
                blured_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
            if len(fingers) == 1:
                center = fingers[0]
                blured_mask = tracking_mask(blured_mask, center[0], center[1], target_size)
        put_string(_mainboard, "Blur Size : ", (_mainboard.shape[1] // 2, 55), target_size, color=(0, 0, 0), size=0.6)
        put_string(_mainboard, "'u' : Blur Size UP ", (_mainboard.shape[1] // 2, 15), color=(255, 0, 0), size=0.6)
        put_string(_mainboard, "'d' : Blur Size DOWN ", (_mainboard.shape[1] // 2, 35), color=(0, 0, 255), size=0.6)
    #샤프닝 모드
    elif mode == 4: 
        if sub_frame is not None: #토글
            frame, fingers = tracking_color(sub_frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size, target_frame=frame)
        else:
            frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size)
            if sharped_mask is None: #마스크가 없을 경우 초기화
                sharped_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
            if len(fingers) == 1:
                center = fingers[0]
                sharped_mask = tracking_mask(sharped_mask, center[0], center[1], target_size)
        put_string(_mainboard, "sharp Size : ", (_mainboard.shape[1] // 2, 55), target_size, color=(0, 0, 0), size=0.6)

    elif not toggle and mode == 5: #지우개 모드
        frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size)
        if len(fingers) == 1:
            center = fingers[0]
            blured_mask, sharped_mask = eraser_mask(blured_mask, sharped_mask, center[0], center[1], target_size)
        put_string(_mainboard, "eraser Size : ", (_mainboard.shape[1] // 2, 55), target_size, color=(0, 0, 0), size=0.6)

    elif mode == 6: #카툰 렌더링 모드
        if sub_frame is not None: #토글
            frame, fingers = tracking_color(sub_frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size, target_frame=frame)
        else:
            frame, fingers = tracking_color(frame.copy(), fingers, lower_green, upper_green, initial_radius=target_size)
            if canny_mask is None: #마스크가 없을 경우 초기화
                canny_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
            if len(fingers) == 1:
                center = fingers[0]
                canny_mask = tracking_mask(canny_mask, center[0], center[1], target_size)
        put_string(_mainboard, "sharp Size : ", (_mainboard.shape[1] // 2, 55), target_size, color=(0, 0, 0), size=0.6)
    
    elif mode == 7: #원근감 보정 코드
        if sub_frame is not None: #토글
            if len(dots) > 0: #점이 있을 시에 그려주기
                for dot in dots:
                    cv2.circle(frame, dot,  2, (0, 255, 0), 2)
            frame, fingers = tracking_color(sub_frame.copy(), fingers, lower_green, upper_green, initial_radius=2, target_frame=frame)

            if len(dots) < 4 and key == 32: #스페이스바 클릭 시에 점 추가
                if len(fingers) == 1:
                    dots.append(fingers[0])
            elif len(dots) > 0 and key == 8: #백스페이스바 클릭 시에 점 삭제
                dots.pop() 
            elif len(dots) == 4 and key == ord('p'): #원근법을 적용할 점이 4개이고, 'p'키를 다시 눌렀을 경우 원근감 적용
                test_img = apply_perspective(frame, dots)
                cv2.imshow("test perspective", test_img)
                dots.clear() #초기화 해주기
                
        put_string(_mainboard, "dot = ", (_mainboard.shape[1] // 2, 55), len(dots), color=(0, 0, 0), size=0.6)
    ###################
    
    
    #현재 프레임이 실시간 영상 송출일 경우
    if not toggle: #토글이 아닐 경우에만 각종 마스크 사용 모드 
        if canny_mask is not None: #카툰 렌더링 적용
            frame = apply_canny(frame.copy(), canny_mask)
        if blured_mask is not None: #블러 적용
            blured_mask = cv2.resize(blured_mask, (frame.shape[1], frame.shape[0]))
            target_frame = cv2.bitwise_and(frame.copy(), frame.copy(), mask=blured_mask)
            blured_frame = cv2.GaussianBlur(target_frame, (15, 15), 0)
            frame[blured_mask > 0] = blured_frame[blured_mask > 0]
        if sharped_mask is not None: #샤프닝 적용
            frame = apply_sharpening(0, frame.copy(), sharped_mask)
    else: #토글일 경우에만 
        pass
    
    #화면 이동에 따른 관심 구역 설정
    move_frame = frame[move_y:move_y + frame_height, move_x:move_x + frame_width]

    if (not toggle) and key == ord('c'): #현재 화면 캡쳐하기
        capture_list.append(move_frame)
    if toggle: #만약 토글된 화면이라면 현재 수정된 이미지를 임시 리스트에 넣기 위한 코드
        temp_list.append(temp_frame) #현재 작업중인 프레임을 임시 리스트에 집어넣기
    if len(capture_list) == 0 and toggle: #만약 토글된 화면이라면 현재 수정된 이미지를 임시 리스트에 넣기 위한 코드
        temp_list.clear()
        toggle = False
    
            
    # _mainboard 중앙에 move_frame을 배치하기 위한 계산
    _mainboard_center_x = _mainboard.shape[1] // 2
    _mainboard_center_y = _mainboard.shape[0] // 2
    frame_center_x = move_frame.shape[1] // 2
    frame_center_y = move_frame.shape[0] // 2

    # _mainboard 중앙에 frame을 배치하기 위한 시작 좌표 계산
    start_x = _mainboard_center_x - frame_center_x
    start_y = _mainboard_center_y - frame_center_y

    # 시작 좌표들이 음수일 경우, 최소값을 0으로 설정
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0

    # _mainboard 중앙에 frame 배치
    end_y = min(start_y + move_frame.shape[0], _mainboard.shape[0])
    end_x = min(start_x + move_frame.shape[1], _mainboard.shape[1])

    #프로그램 관련 설명, 설정 텍스트 출력
    put_string(_mainboard, "distance : ", (180, 15), distance, color=(0,0,0), size=0.6) 
    put_string(_mainboard, "mode : ", (180, 35), mode_name[mode], color=(255, 0,0), size=0.6)
    put_string(_mainboard, "current_scale : ", (180, 55), round(current_scale, 2), color=(0, 0, 255), size=0.6)

    put_string(_mainboard, "<Keyboard>", (10, 60), color=(0,0,0))
    put_string(_mainboard, "'ESC' : EXIT", (10, 90), color=(0,0,0))
    put_string(_mainboard, "'o' : Common", (10, 120), color=(0,0,0))
    put_string(_mainboard, "'r' : Reset", (10, 150), color=(0,0,0))
    put_string(_mainboard, "'z' : Zoom", (10, 180), color=(0,0,0))
    put_string(_mainboard, "'m' : Move", (10, 210), color=(0,0,0))
    put_string(_mainboard, "'b' : Blur", (10, 230), color=(0,0,0))
    put_string(_mainboard, "'s' : Sharp", (10, 250), color=(0,0,0))
    put_string(_mainboard, "'e' : Eraser", (10, 270), color=(0,0,0))
    put_string(_mainboard, "'k' : cartoon", (10, 290), color=(0,0,0))
    put_string(_mainboard, "'p' : perspective", (10, 310), color=(0,0,0))

    put_string(_mainboard, "Temp_List = ", (40, main_height - 40), len(temp_list), color=(0,0,255), size=0.7)
    put_string(_mainboard, "Captrue_count = ", (main_width // 2, main_height - 40), len(capture_list), color=(0,0,255), size=0.7)
    put_string(_mainboard, "toggle = ", (main_width // 2, main_height - 20), toggle, color=(0,0,255), size=0.7)
    
    #메인보드에 수정된 영상 붙이기
    _mainboard[start_y:end_y, start_x:end_x] = move_frame
    _mainboard[0:main_height, (main_width - frame_width)//2 + frame_width:main_width] = _sideboard
    _programboard[0:main_height, 0:main_width] = _mainboard
    _programboard[0:main_height, main_width:main_width + _resultboard.shape[1]] = _resultboard

    cv2.imshow(title, _programboard)

capture.release()
cv2.destroyAllWindows()