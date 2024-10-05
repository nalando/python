import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Mesh 模組
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# 開啟攝像頭
cap = cv2.VideoCapture(0)  # 0 是默認的攝像頭，如果是USB攝像頭，可能是 1 或其他編號

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("未能捕捉到影像")
        break

    # 將影像轉為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 進行臉部偵測
    results = face_mesh.process(image_rgb)
    
    # 如果偵測到臉部
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 繪製臉部網格和五官
            mp_drawing.draw_landmarks(
                image, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    # 顯示影像
    cv2.imshow('Face Mesh', image)

    # 按下 'q' 鍵退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Mesh 模組
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# 開啟攝像頭
cap = cv2.VideoCapture(0)  # 0 是默認的攝像頭，如果是USB攝像頭，可能是 1 或其他編號

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("未能捕捉到影像")
        break

    # 將影像轉為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 進行臉部偵測
    results = face_mesh.process(image_rgb)
    
    # 如果偵測到臉部
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 繪製臉部網格和五官
            mp_drawing.draw_landmarks(
                image, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    # 顯示影像
    cv2.imshow('Face Mesh', image)

    # 按下 'q' 鍵退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Mesh 模組
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# 開啟攝像頭
cap = cv2.VideoCapture(0)  # 0 是默認的攝像頭，如果是USB攝像頭，可能是 1 或其他編號

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("未能捕捉到影像")
        break

    # 將影像轉為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 進行臉部偵測
    results = face_mesh.process(image_rgb)
    
    # 如果偵測到臉部
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 繪製臉部網格和五官
            mp_drawing.draw_landmarks(
                image, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    # 顯示影像
    cv2.imshow('Face Mesh', image)

    # 按下 'q' 鍵退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
