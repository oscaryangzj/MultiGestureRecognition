import cv2
import mediapipe as mp
import numpy as np

from gesture.config import project_path


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def create_hand_landmarker(config, running_mode):
    options_cfg = config["hand_landmarker"]
    model_path = project_path(options_cfg["model_asset_path"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model is missing at {model_path}. "
            "Run scripts/download_hand_landmarker.py first."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate[options_cfg["delegate"]],
        ),
        running_mode=running_mode,
        num_hands=int(options_cfg["num_hands"]),
        min_hand_detection_confidence=float(options_cfg["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(options_cfg["min_hand_presence_confidence"]),
        min_tracking_confidence=float(options_cfg["min_tracking_confidence"]),
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_video_frame(landmarker, bgr_frame, timestamp_ms):
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(image, int(timestamp_ms))
    if not result.hand_landmarks:
        return None
    return np.asarray(
        [[point.x, point.y] for point in result.hand_landmarks[0]],
        dtype=np.float32,
    )


def draw_hand_landmarks(frame, points):
    if points is None:
        return frame
    height, width = frame.shape[:2]
    xy = [(int(point[0] * width), int(point[1] * height)) for point in points]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, xy[start], xy[end], (60, 220, 60), 2, cv2.LINE_AA)
    for point in xy:
        cv2.circle(frame, point, 3, (0, 120, 255), -1, cv2.LINE_AA)
    return frame
