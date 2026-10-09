import numpy as np


def _scale_points(points, image_size):
    width, height = image_size
    result = np.asarray(points, dtype=np.float64).copy()
    result[..., 0] *= width / height
    return result


def direction_trace(session, interval, config):
    """Return the representative trajectory and fitted PCA projection for UI review."""
    settings = config["annotation"]["periodic"]
    start, end = int(interval["start_frame"]), int(interval["end_frame"])
    frames = np.flatnonzero(session["detected"][start:end + 1]) + start
    if len(frames) < 4:
        return None
    blocks = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
    block = max(blocks, key=len)
    if len(block) < 4:
        return None
    points = np.asarray(session["points"])
    trajectory = _scale_points(points[block][:, settings["fingertip_indices"]], session["image_size"]).mean(axis=1)
    center = trajectory.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((trajectory - center).T))
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis *= -1
    angle = abs(float(np.degrees(np.arctan2(axis[1], axis[0]))))
    projected = (trajectory - center) @ axis
    smooth = int(settings["smooth_frames"])
    if smooth > 1:
        prefix = np.r_[0.0, np.cumsum(projected)]
        projected = np.asarray([(prefix[min(index + 1, len(projected))]
                                 - prefix[max(0, index + 1 - smooth)]) / min(smooth, index + 1)
                                for index in range(len(projected))])
    return {"frames": block, "trajectory": trajectory, "projection": projected,
            "axis": axis, "angle_degrees": angle,
            "eligible": angle <= float(settings["max_axis_angle_degrees"])}


def propose_direction_events(session, waving_interval, config):
    """Make editable offline reversal drafts inside a reviewed waving interval."""
    settings = config["annotation"]["periodic"]
    fingertips = tuple(settings["fingertip_indices"])
    start, end = int(waving_interval["start_frame"]), int(waving_interval["end_frame"])
    points = np.asarray(session["points"])
    times = session["times"]
    detected = session["detected"]
    image_size = session["image_size"]
    palm = _scale_points(points[:, [0, 9]], image_size)
    fingertips = _scale_points(points[:, fingertips], image_size).mean(axis=1)
    scale = np.linalg.norm(palm[:, 1] - palm[:, 0], axis=1)

    drafts = []
    valid_frames = np.flatnonzero(detected[start:end + 1]) + start
    if len(valid_frames) < max(4, int(settings["smooth_frames"]) + 2):
        return drafts
    for block in np.split(valid_frames, np.flatnonzero(np.diff(valid_frames) > 1) + 1):
        if len(block) < max(4, int(settings["smooth_frames"]) + 2):
            continue
        trajectory = fingertips[block]
        center = trajectory.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((trajectory - center).T))
        axis = vectors[:, -1]
        if axis[0] < 0:
            axis *= -1
        angle = abs(np.degrees(np.arctan2(axis[1], axis[0])))
        if angle > float(settings["max_axis_angle_degrees"]):
            continue
        projected = (trajectory - center) @ axis
        smooth = int(settings["smooth_frames"])
        if smooth > 1:
            prefix = np.r_[0.0, np.cumsum(projected)]
            projected = np.asarray([
                (prefix[min(index + 1, len(projected))] - prefix[max(0, index + 1 - smooth)])
                / min(smooth, index + 1)
                for index in range(len(projected))
            ])
        local_scale = float(np.median(scale[block]))
        if not np.isfinite(local_scale) or local_scale <= 0:
            continue
        minimum = float(settings["min_leg_scale"]) * local_scale
        reversal = float(settings["reverse_scale"]) * local_scale
        confirm = int(settings["reverse_confirm_frames"])
        anchor_index, anchor = 0, float(projected[0])
        direction, extremum_index, extremum = 0, 0, anchor
        reverse_count = 0
        for index in range(1, len(projected)):
            value = float(projected[index])
            if direction == 0:
                delta = value - anchor
                if abs(delta) >= minimum:
                    direction = 1 if delta > 0 else -1
                    extremum_index, extremum = index, value
                continue
            advanced = value > extremum if direction > 0 else value < extremum
            if advanced:
                extremum_index, extremum = index, value
                reverse_count = 0
                continue
            reverse_distance = extremum - value if direction > 0 else value - extremum
            if reverse_distance >= reversal:
                reverse_count += 1
            else:
                reverse_count = 0
            if reverse_count < confirm:
                continue
            start_frame = int(block[anchor_index])
            end_frame = int(block[index])
            drafts.append({
                "prompt_index": waving_interval.get("prompt_index"),
                "label": "left_to_right" if direction > 0 else "right_to_left",
                "start_frame": start_frame,
                "extremum_frame": int(block[extremum_index]),
                "end_frame": end_frame,
            })
            anchor_index, anchor = extremum_index, extremum
            direction, reverse_count = 0, 0
    return drafts
