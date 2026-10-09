import zipfile
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from gesture.config import project_path
from gesture.official_gesture_classifier import OfficialGestureClassifier


@dataclass
class StateRecognizer:
    landmarker: object
    classifier: OfficialGestureClassifier
    category_indices: list


@contextmanager
def create_state_recognizer(config, classes=None):
    model_path = project_path(config["gesture_recognizer"]["model_asset_path"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Gesture Recognizer model is missing at {model_path}. "
            "Run python -m scripts.download_gesture_recognizer first."
        )
    hand_config = config["hand_landmarker"]
    categories = config["gesture_recognizer"]["category_names"]
    with zipfile.ZipFile(model_path) as bundle:
        hand_model = bundle.read("hand_landmarker.task")
        gesture_model = bundle.read("hand_gesture_recognizer.task")
    settings = config["gesture_recognizer"]
    classifier = OfficialGestureClassifier(gesture_model, int(settings["classifier_threads"]),
                                          float(settings["normalization_epsilon"]))
    labels = config["labels"]["state_classes"] if classes is None else classes
    indices = [classifier.categories.index(categories[label]) for label in labels]
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=hand_model,
            delegate=mp.tasks.BaseOptions.Delegate[hand_config["delegate"]],
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=int(hand_config["num_hands"]),
        min_hand_detection_confidence=float(hand_config["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(hand_config["min_hand_presence_confidence"]),
        min_tracking_confidence=float(hand_config["min_tracking_confidence"]),
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        yield StateRecognizer(landmarker, classifier, indices)


def recognize_video_frame(recognizers, bgr_frame, timestamp_ms, include_world_landmarks=False):
    """Extract the hand once; return original class scores and optional world points."""
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = recognizers.landmarker.detect_for_video(image, int(timestamp_ms))
    if not result.hand_landmarks:
        return (None, None, None) if include_world_landmarks else (None, None)
    points = np.asarray(
        [[point.x, point.y] for point in result.hand_landmarks[0]],
        dtype=np.float32,
    )
    all_scores = recognizers.classifier.predict(result, (bgr_frame.shape[1], bgr_frame.shape[0]))
    probabilities = all_scores[recognizers.category_indices]
    if include_world_landmarks:
        world_points = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]],
            dtype=np.float32,
        )
        return points, probabilities, world_points
    return points, probabilities
