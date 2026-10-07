import numpy as np


NUM_LANDMARKS = 21
FEATURE_SIZE = NUM_LANDMARKS * 2
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def normalize_landmarks(points, min_hand_scale, local_scale):
    """Return wrist centered, palm-scale-normalized x/y features, or None."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (NUM_LANDMARKS, 2) or not np.isfinite(points).all():
        return None

    centered = points - points[WRIST]
    if local_scale != "wrist_to_middle_mcp":
        raise ValueError(f"Unsupported local_scale: {local_scale}")
    scale = float(np.linalg.norm(centered[MIDDLE_MCP]))
    if scale <= float(min_hand_scale):
        return None
    return (centered / scale).reshape(-1).astype(np.float32)
