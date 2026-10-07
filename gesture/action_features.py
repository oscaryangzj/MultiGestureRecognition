import numpy as np

from gesture.features import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST, normalize_landmarks


ACTION_FEATURE_SIZE = 49
ACTION_MIRROR_MOTION_INDICES = [42, 44, 46]  # camera vx, ax, angular velocity


def action_feature_size(config):
    feature_config = config["features"]
    base_size = ACTION_FEATURE_SIZE - (0 if feature_config.get("action_include_acceleration", True) else 2)
    angle_size = (len(feature_config["action_finger_angle_triples"])
                  if feature_config.get("action_include_finger_angles", False) else 0)
    return base_size + angle_size


def action_mirror_motion_indices(config):
    if config["features"].get("action_include_acceleration", True):
        return ACTION_MIRROR_MOTION_INDICES
    return [42, 44]  # camera vx and angular velocity without acceleration


class ActionFeatureExtractor:
    """Causal features; missing frames are skipped and long gaps reset history."""

    def __init__(self, config):
        self.config = config
        self.missing = 0
        self._clear_history()

    def _clear_history(self):
        self.previous_time = None
        self.previous_palm = None
        self.previous_velocity = None
        self.previous_angle = None
        self.unwrapped_angle = None

    def update(self, points, state_scores, elapsed, image_size=(1, 1), world_points=None):
        feature_config = self.config["features"]
        geometry = None if points is None else np.asarray(points, dtype=np.float64).copy()
        align = feature_config.get("action_align_palm_axis", False)
        if geometry is not None and align:
            # Image-normalized x/y have different units on a non-square image.
            geometry[:, 0] *= image_size[0] / image_size[1]
        local = None if geometry is None else normalize_landmarks(
            geometry, feature_config["min_hand_scale"], feature_config["local_scale"])
        if local is None or state_scores is None:
            self.missing += 1
            reset = self.missing == int(self.config["data"]["no_hand_reset_frames"])
            if reset:
                self._clear_history()
            return None, reset
        finger_angles = []
        if feature_config.get("action_include_finger_angles", False):
            if world_points is None:
                raise ValueError("3D finger angles are enabled; re-extract Action landmarks to create world_landmarks.csv")
            world = np.asarray(world_points, dtype=np.float64)
            if world.shape != (21, 3):
                raise ValueError("World landmarks must have shape [21, 3]")
            for a, b, c in feature_config["action_finger_angle_triples"]:
                first = world[b] - world[a]
                second = world[c] - world[b]
                finger_angles.append(np.arctan2(np.linalg.norm(np.cross(first, second)), np.dot(first, second)))
        self.missing = 0
        palm = np.asarray(points)[feature_config["palm_landmark_indices"]].mean(axis=0)
        direction = geometry[MIDDLE_MCP] - geometry[WRIST]
        angle = float(np.arctan2(direction[1], direction[0]))
        if align:
            axis = direction / np.linalg.norm(direction)
            # Align wrist -> middle MCP with image-up, without rotating motion inputs.
            rotation = np.asarray([[-axis[1], axis[0]], [-axis[0], -axis[1]]])
            local = (local.reshape(21, 2) @ rotation.T).astype(np.float32)
            # Use the palm's index-to-pinky side, independent of finger flexion.
            if feature_config.get("action_canonicalize_hand", False) and local[INDEX_MCP, 0] < local[PINKY_MCP, 0]:
                local[:, 0] *= -1
            local[WRIST] = [0, 0]
            local[MIDDLE_MCP] = [0, -1]
            local = local.reshape(-1)
        velocity = np.zeros(2, np.float32)
        acceleration = np.zeros(2, np.float32)
        angular_velocity = 0.0
        had_history = self.previous_time is not None
        if not had_history:
            self.unwrapped_angle = angle
        else:
            dt = float(elapsed) - self.previous_time
            if dt <= 0:
                raise ValueError("Action feature times must strictly increase")
            velocity = (palm - self.previous_palm) / dt
            if self.previous_velocity is not None:
                acceleration = (velocity - self.previous_velocity) / dt
            delta = angle - self.previous_angle
            delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
            self.unwrapped_angle += delta
            angular_velocity = delta / dt
        self.previous_time = float(elapsed)
        self.previous_palm = palm.copy()
        self.previous_velocity = velocity.copy() if had_history else None
        self.previous_angle = angle
        scores = [state_scores[name] for name in feature_config["action_state_classes"]]
        wrist = (angular_velocity if feature_config.get("action_wrist_feature", "angle") == "angular_velocity"
                 else self.unwrapped_angle)
        motion = (velocity, acceleration) if feature_config.get("action_include_acceleration", True) else (velocity,)
        features = np.concatenate((local, *motion, [wrist], scores, finger_angles)).astype(np.float32)
        if len(features) != action_feature_size(self.config):
            raise ValueError("Action features require 21 x/y points and two State scores")
        return features, False


def make_action_features(points_by_frame, detected, scores, times, config, image_size=(1, 1), input_regions=None,
                         world_points_by_frame=None):
    features = np.zeros((len(times), action_feature_size(config)), np.float32)
    valid = np.zeros(len(times), bool)
    segments = np.zeros(len(times), np.int64)
    extractor = ActionFeatureExtractor(config)
    segment = 0
    previous_region = None
    for index, elapsed in enumerate(times):
        if input_regions is not None:
            region = int(input_regions[index])
            if region < 0:
                segments[index] = -1
                continue
            if region != previous_region:
                if previous_region is not None:
                    segment += 1
                extractor = ActionFeatureExtractor(config)
                previous_region = region
        points = points_by_frame[index] if detected[index] else None
        state_scores = scores[index] if detected[index] else None
        world_points = world_points_by_frame[index] if world_points_by_frame is not None and detected[index] else None
        feature, reset = extractor.update(points, state_scores, elapsed, image_size, world_points)
        if reset:
            segment += 1
        segments[index] = segment
        if feature is not None:
            features[index] = feature
            valid[index] = True
    return features, valid, segments
