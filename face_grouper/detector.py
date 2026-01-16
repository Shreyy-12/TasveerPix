import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .config import FACE_SIZE


import face_recognition
import numpy as np
import cv2


def detect_faces(image):
    """
    Detect faces using dlib (face_recognition).
    Returns list of bounding boxes.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    return boxes


def extract_face_embedding(image):
    """
    Detects face and extracts 128D embedding using dlib.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    if len(boxes) == 0:
        return None

    encodings = face_recognition.face_encodings(rgb, [boxes[0]])
    if len(encodings) == 0:
        return None

    return encodings[0]




def crop_face(face, image, size=(112, 112)):
    try:
        x1, y1, x2, y2 = map(int, face.bbox)

        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        return cv2.resize(face_crop, size)

    except Exception as e:
        print(f"⚠️ Error cropping face: {e}")
        return None
# def crop_face(face, image):
#     x1, y1, x2, y2 = map(int, face.bbox)
#     return cv2.resize(image[y1:y2, x1:x2], FACE_SIZE)