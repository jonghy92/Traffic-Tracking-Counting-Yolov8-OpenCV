# 라이브러리
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import * # tracker.py 모듈 로드
import sys # sys 모듈 로드

# sys - 터미널로 .py 실행시 video_path & 명 - 매개변수 전달
# ex. 터미널 실행 명령어 : python main.py ./video_data/test_video3.mp4
try:
    video_path = str(sys.argv[1])
except:
    print('\n')
    print("Error : 비디오 파일의 경로를 실행어 뒤에 같이 넣어주세요.")
    print("Ex. python main.py ./video_data/test_video3.mp4")
    print('\n')


# YOLO Model 가져오기
model = YOLO('./models/yolov8n.pt')

# COCO Data Set 레이블 리스트
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Object 트래킹
tracker = Tracker()



# 비디오 읽기 및 정보 파악
cap = cv2.VideoCapture(f'{video_path}') #cap = cv2.VideoCapture('./video_data/test_video2.mp4')
origin_w, origin_h, origin_fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("./video_detection_results/detection_video_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), origin_fps, (origin_w, origin_h))

# 터미널 비디오 정보 출력
print("# ================================================= #")
print("비디오 넓이 (w) : ", origin_w)
print("비디오 높이 (h) : ", origin_h)
print("비디오 FPS : ", origin_fps)
print("# ================================================= #")

count=0
vehicle_passed_list = []
down={}
counter_down=set()

# 비디오 플레이 - frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # Yolo 검출
    results = model.predict(frame)

    extracted_data = results[0].boxes.data
    extracted_data = extracted_data.detach().cpu().numpy()
    df_extracted = pd.DataFrame(extracted_data).astype("float")


    list = []

    for inex, row in df_extracted.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        if 'car' in c:
            list.append([x1,y1,x2,y2, "car"])
        elif 'motorcycle' in c:
            list.append([x1,y1,x2,y2, "motorcycle"])
        # elif 'bus' in c:
        #     list.append([x1,y1,x2,y2])
        # elif 'truck' in c:
        #     list.append([x1,y1,x2,y2])


    # 자동차 오브젝트 트랙킹 부분
    bbox_id, yolo_detect = tracker.update(list)

    # 자동차 오브젝트 센터점 선별
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2 # cx = Center X
        cy=int(y3+y4)//2 # cy = Center Y


        #----------------------- Condition -----------------------#
        # 빨간선 구축
        y = int(origin_h/2) # <-- red line y position
        offset = 7

        # 빨간선 조건문 : 선에 닿을때마다 카운트 진행
        if y < (cy + offset) and y > (cy - offset):
        
            down[id] = cy # cy : 현재 포지션 (ojbect 중앙점)

            if id in down:
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2) # 바운딩 박스 - 자동차
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                counter_down.add(id)
                
                # 부가정보 리스트화
                vehicle_passed_list.append([id, yolo_detect, (x3,y3,x4,y4), frame])
        #--------------------- Condition End ---------------------#

    # Line Horizontal
    text_color = (255, 255, 255)
    red_color = (0, 0, 255)

    # Draw Line & Text
    cv2.line(frame,(0, int(origin_h/2)), (origin_w, int(origin_h/2)), red_color,3) # automate test_video3
    cv2.putText(frame,('Counting Line'),(0, int(origin_h/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    # Draw Counter Vehicles Dash Board
    cv2.putText(frame, ('Pass Traffic Counted : ' + str(len(counter_down))), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 1, cv2.LINE_AA)

    # ESC Exit 구축
    cv2.imshow("frames", frame)
    if cv2.waitKey(1)&0xFF==27: # ESC Exit
        break

    # 검출된 비디오 저장
    video_writer.write(frame)

cap.release()
video_writer.release() 
cv2.destroyAllWindows()


# 컴출된 데이터 프레임화 및 저장
df_vehicle_detected = pd.DataFrame(vehicle_passed_list,
                    columns=["id", "yolo_detected_rslt", "bounding_box", "frame_view"])
df_vehicle_detected.to_csv('./video_detection_results/df_vehicle_detected.csv', header=True)



# 터미널 실행결과 출력
print('\n\n\n')
print("# ================================================= #")
print('\n')
print("Detection Video Generated : ./video_detection_results/detection_video_output.mp4 ...! ")
print("Detection Data Saved : ./video_detection_results/detection_data_output.csv")
print('\n')
print("Number of Vehicles Passed : ", df_vehicle_detected["id"].nunique()) # 지나간 차량 횟수
print("Vehicles' Type Passed : ", df_vehicle_detected['yolo_detected_rslt'].unique()) # pass된 운송 수단 종류
print('\n')
print("Finished ...!")
print('\n')
print("# ================================================= #")
