import numpy as np

# 比较人脸
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1.2):
    if len(known_face_encodings) == 0:
        return [], np.empty((0))
    dis = np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1)
    return list(dis <= tolerance), dis