import io
import zipfile

import numpy as np
from ai_edge_litert.interpreter import Interpreter


class OfficialGestureClassifier:
    """Run Google's bundled gesture models on already detected hand landmarks.

    Preprocessing follows MediaPipe v0.10.35's LandmarksToMatrixCalculator and
    HandednessToMatrixCalculator, with the default unrotated VIDEO input.
    """

    def __init__(self, model_bundle, threads, epsilon):
        with zipfile.ZipFile(io.BytesIO(model_bundle)) as bundle:
            embedder = bundle.read("gesture_embedder.tflite")
            classifier = bundle.read("canned_gesture_classifier.tflite")
        with zipfile.ZipFile(io.BytesIO(classifier)) as metadata:
            self.categories = metadata.read("labels.txt").decode().splitlines()
        self.embedder = Interpreter(model_content=embedder, num_threads=threads)
        self.classifier = Interpreter(model_content=classifier, num_threads=threads)
        self.embedder.allocate_tensors()
        self.classifier.allocate_tensors()
        self.inputs = {item["name"]: item["index"] for item in self.embedder.get_input_details()}
        self.embedding_output = self.embedder.get_output_details()[0]["index"]
        self.classifier_input = self.classifier.get_input_details()[0]["index"]
        self.classifier_output = self.classifier.get_output_details()[0]["index"]
        self.epsilon = epsilon

    def normalize(self, landmarks, image_size=None):
        # Match MediaPipe LandmarksToMatrixCalculator: aspect ratio, wrist origin,
        # then the maximum x/y extent. VIDEO inputs have no image rotation.
        points = np.asarray([[point.x, point.y, point.z] for point in landmarks], np.float32)
        if image_size is not None:
            width, height = image_size
            points[:, :2] *= np.asarray([width, height], np.float32) / max(width, height)
        points -= points[0].copy()
        scale = np.ptp(points[:, :2], axis=0).max() + self.epsilon
        return (points / scale)[None].astype(np.float32)

    def predict(self, result, image_size):
        handedness = result.handedness[0][0]
        right_score = handedness.score if handedness.category_name == "Right" else 1 - handedness.score
        self.embedder.set_tensor(self.inputs["hand"], self.normalize(result.hand_landmarks[0], image_size))
        self.embedder.set_tensor(self.inputs["world_hand"], self.normalize(result.hand_world_landmarks[0]))
        self.embedder.set_tensor(self.inputs["handedness"], np.asarray([[right_score]], np.float32))
        self.embedder.invoke()
        self.classifier.set_tensor(self.classifier_input, self.embedder.get_tensor(self.embedding_output))
        self.classifier.invoke()
        return self.classifier.get_tensor(self.classifier_output)[0]
