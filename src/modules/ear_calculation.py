import numpy as np

def calculate_ear(landmarks_eye):
    """
    Calcula o EAR (Eye Aspect Ratio) a partir dos pontos do olho.
    """
    vertical_distances_1 = np.linalg.norm(landmarks_eye[1] - landmarks_eye[5])
    vertical_distances_2 = np.linalg.norm(landmarks_eye[2] - landmarks_eye[4])
    horizontal_distance = np.linalg.norm(landmarks_eye[0] - landmarks_eye[3])
    
    ear = (vertical_distances_1 + vertical_distances_2) / (2.0 * horizontal_distance)
    return ear