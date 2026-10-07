from contextlib import ExitStack, contextmanager

import cv2
import mediapipe as mp
import numpy as np

from gesture.config import project_path


@contextmanager
def create_state_recognizer(config):
    model_path = project_path(config["gesture_recognizer"]["model_asset_path"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Gesture Recognizer model is missing at {model_path}. "
            "Run python -m scripts.download_gesture_recognizer first."
        )
    hand_config = config["hand_landmarker"]
    categories = config["gesture_recognizer"]["category_names"]
    with ExitStack() as stack:
        recognizers = []
        for label in config["labels"]["state_classes"]:
            category = categories[label]
            options = mp.tasks.vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=mp.tasks.BaseOptions.Delegate[hand_config["delegate"]],
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=int(hand_config["num_hands"]),
                min_hand_detection_confidence=float(hand_config["min_hand_detection_confidence"]),
                min_hand_presence_confidence=float(hand_config["min_hand_presence_confidence"]),
                min_tracking_confidence=float(hand_config["min_tracking_confidence"]),
                canned_gesture_classifier_options=mp.tasks.components.processors.ClassifierOptions(
                    category_allowlist=[category], score_threshold=0.0,
                ),
            )
            recognizer = stack.enter_context(
                mp.tasks.vision.GestureRecognizer.create_from_options(options)
            )
            recognizers.append((category, recognizer))
        yield recognizers


def recognize_video_frame(recognizers, bgr_frame, timestamp_ms, include_world_landmarks=False):
    """Return [21, 2] landmarks and scores, optionally with [21, 3] world points.

    The official task returns only the winning category, even with max_results=-1.
    Separate allowlists expose each target's original score without renormalizing
    the three targets or replacing the unreturned scores with artificial zeros.
    This runs one official task per target class, including its hand tracker.
    """
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Every tracker consumes every frame, including frames with no detected hand.
    results = [
        recognizer.recognize_for_video(image, int(timestamp_ms))
        for _category, recognizer in recognizers
    ]
    if any(not result.hand_landmarks for result in results):
        return (None, None, None) if include_world_landmarks else (None, None)
    points = np.asarray(
        [[point.x, point.y] for point in results[0].hand_landmarks[0]],
        dtype=np.float32,
    )
    probabilities = np.zeros(len(recognizers), dtype=np.float32)
    for index, ((category, _recognizer), result) in enumerate(zip(recognizers, results)):
        probabilities[index] = next(
            (item.score for item in result.gestures[0] if item.category_name == category),
            0.0,
        )
    if include_world_landmarks:
        world_points = np.asarray(
            [[point.x, point.y, point.z] for point in results[0].hand_world_landmarks[0]],
            dtype=np.float32,
        )
        return points, probabilities, world_points
    return points, probabilities
