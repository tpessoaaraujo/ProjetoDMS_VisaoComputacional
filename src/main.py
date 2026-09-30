import cv2
import mediapipe as mp
import numpy as np
import time
from modules.ear_calculation import calculate_ear
from modules.landmark_utils import extract_points

# Parâmetros de calibração do EAR para detecção
EAR_THRESHOLD = 0.22  # Valor de EAR abaixo do qual o olho é considerado fechado
CONSECUTIVE_FRAMES = 50  # Quadros consecutivos para considerar o olho fechado
CALIBRATION_TIME = 5  # Tempo de calibração em segundos do EAR

# Configurações iniciais do MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicia o modelo Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define os índices dos pontos de referência dos olhos na malha facial
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Inicializa o objeto de captura de vídeo (0 indica a câmera padrão)
cap = cv2.VideoCapture(0)

# Variáveis da aplicação
counter = 0
eye_closed_start = None
eye_closed_duration = 0.0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Falha ao capturar o vídeo.")
        break
    
    # Converte a imagem de BGR do OpenCV para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Processa a imagem para detectar a malha facial
    results = face_mesh.process(rgb_frame)
    
    # Coverte a imagem para exibição em tons de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        height, width, _ = frame.shape # Retorna a altura e largura do frame
        
        left_eye = extract_points(LEFT_EYE_INDICES, face_landmarks, width, height)
        right_eye = extract_points(RIGHT_EYE_INDICES, face_landmarks, width, height)
        
        # Calcula o EAR para ambos os olhos
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear_average = (left_ear + right_ear) / 2.0
        
        if ear_average < EAR_THRESHOLD:
            counter += 1
            
            if eye_closed_start is None:
                eye_closed_start = time.perf_counter()
            eye_closed_duration = time.perf_counter() - eye_closed_start
            
        else:
            counter = 0
            eye_closed_start = None
            eye_closed_duration = 0.0
        
        # Desenha a malha facial completa na imagem    
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(195, 232, 46), thickness=1)
        )
        
        # Seleciona apenas as conexões dos olhos e desenha na imagem
        eye_connections = (mp_face_mesh.FACEMESH_LEFT_EYE | mp_face_mesh.FACEMESH_RIGHT_EYE)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=eye_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(25, 25, 255), thickness=2)
        )
        
        # Exibe o EAR na tela
        cv2.putText(frame, f"EAR: {ear_average:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"'ESC' para sair", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Se o contador atingir o limite de quadros consecutivos, exibe um alerta
        if counter >= CONSECUTIVE_FRAMES:
            cv2.putText(frame, "ALERTA: SONO DETECTADO", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 25, 255), 2)
            cv2.putText(frame, f"Tempo de olhos fechados: {eye_closed_duration:.2f}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 25, 255), 2)
    
    cv2.namedWindow("Monitoramento DMS", cv2.WINDOW_NORMAL)
    cv2.imshow("Monitoramento DMS", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
# Libera a câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()