import numpy as np

def extract_points(indices, face_landmarks, width, height):
    """
    Extrai os pontos do rosto com base nos índices e converte para coordenadas de pixel.
    """
    points = []
    
    for index in indices:
        landmark = face_landmarks.landmark[index]
        
        points.append([
            int(landmark.x * width),
            int(landmark.y * height)
        ])
        
    return np.array(points)