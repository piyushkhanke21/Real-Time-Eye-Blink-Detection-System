"""Face landmark backends. Returns the 6 EAR points for each eye in pixel coordinates.

Works with both MediaPipe APIs:
  * legacy  mp.solutions.face_mesh   (mediapipe <= ~0.10.21, bundled model)
  * tasks   FaceLandmarker           (newer mediapipe, model auto-downloaded once)
"""
import os
import urllib.request

import numpy as np

# MediaPipe face mesh indices ordered p1..p6 (corner, top, top, corner, bottom, bottom)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MODEL_URL = (r"C:\Users\piyus\Downloads\blink_app\blink_app\models\face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")


def _extract(landmarks, shape):
    h, w = shape[:2]
    pts = lambda idx: np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx], dtype=np.float64)
    return pts(LEFT_EYE), pts(RIGHT_EYE)


class LegacyBackend:
    name = "mediapipe-solutions"

    def __init__(self):
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def find_eyes(self, rgb):
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return _extract(res.multi_face_landmarks[0].landmark, rgb.shape)


class TasksBackend:
    name = "mediapipe-tasks"

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        _ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def find_eyes(self, rgb):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect(image)
        if not res.face_landmarks:
            return None
        return _extract(res.face_landmarks[0], rgb.shape)


def _ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tmp = MODEL_PATH + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, MODEL_PATH)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"could not download face model ({exc}). Download {MODEL_URL} "
                           f"manually and save it as {MODEL_PATH}") from exc


def create_backend():
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return LegacyBackend()
    return TasksBackend()
