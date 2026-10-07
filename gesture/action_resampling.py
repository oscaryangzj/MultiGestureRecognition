import numpy as np

from gesture.action_features import make_action_features


class ActionResampler:
    """Emit a fixed time grid once its right-hand observation has arrived."""

    def __init__(self, sample_rate_hz):
        self.rate = float(sample_rate_hz)
        self.origin = None
        self.index = 0
        self.previous = None

    def update(self, points, scores, elapsed):
        return [sample[:3] for sample in self._update(points, scores, elapsed, None)]

    def update_with_world(self, points, scores, elapsed, world_points):
        return self._update(points, scores, elapsed, world_points)

    def _update(self, points, scores, elapsed, world_points):
        elapsed = float(elapsed)
        if self.origin is None:
            self.origin = elapsed
        current = (elapsed, None if points is None else np.asarray(points), scores,
                   None if world_points is None else np.asarray(world_points))
        result = []
        while self.origin + self.index / self.rate <= elapsed + 1e-9:
            sample_time = self.origin + self.index / self.rate
            sample_points, sample_scores, sample_world = None, None, None
            if abs(sample_time - elapsed) < 1e-9:
                sample_points, sample_scores, sample_world = current[1:]
            elif self.previous is not None:
                before, old_points, old_scores, old_world = self.previous
                if old_points is not None and points is not None and old_scores is not None and scores is not None:
                    fraction = (sample_time - before) / (elapsed - before)
                    sample_points = old_points + fraction * (current[1] - old_points)
                    sample_scores = {name: old_scores[name] + fraction * (scores[name] - old_scores[name])
                                     for name in scores}
                    if old_world is not None and current[3] is not None:
                        sample_world = old_world + fraction * (current[3] - old_world)
            result.append((sample_time, sample_points, sample_scores, sample_world))
            self.index += 1
        self.previous = current
        return result


def resample_action_session(session, config):
    """Keep source video/annotations intact and build the model's time grid."""
    from gesture.action_data import make_action_targets

    resampler = ActionResampler(config["data"]["action_sample_rate_hz"])
    source_regions = session.get("input_regions")
    previous_region = None
    samples, available = [], []
    use_world = config["features"].get("action_include_finger_angles", False)
    for frame, elapsed in enumerate(session["times"]):
        region = int(source_regions[frame]) if source_regions is not None else 0
        if region != previous_region:
            # Keep the session time grid, but never interpolate across a task boundary.
            resampler.previous = None
        previous_region = region
        points = session["points"][frame] if session["detected"][frame] and region >= 0 else None
        scores = session["state_scores"][frame] if points is not None else None
        world_points = session["world_points"][frame] if use_world and points is not None else None
        emitted = (resampler.update_with_world(points, scores, elapsed, world_points) if use_world else
                   [(time, item_points, item_scores, None)
                    for time, item_points, item_scores in resampler.update(points, scores, elapsed)])
        samples.extend(emitted)
        available.extend([elapsed] * len(emitted))
    times = np.asarray([item[0] for item in samples])
    points = [item[1] for item in samples]
    scores = [item[2] for item in samples]
    world_points = [item[3] for item in samples]
    detected = np.asarray([point is not None and score is not None for point, score in zip(points, scores)])
    # An interpolated sample touching an ignored observation cannot gain supervision.
    right = np.searchsorted(session["times"], times, side="left").clip(0, len(session["times"]) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.isclose(times, session["times"][right], atol=1e-9, rtol=0)
    left[exact] = right[exact]
    excluded = session["ignored_frames"][left] | session["ignored_frames"][right]
    input_regions = None
    if source_regions is not None:
        input_regions = np.where(source_regions[left] == source_regions[right], source_regions[right], -1)
        detected &= input_regions >= 0
    features, valid, segments = make_action_features(
        points, detected, scores, times, config, session.get("image_size", (1, 1)), input_regions,
        world_points_by_frame=world_points if use_world else None)
    intervals = []
    for item in session["intervals"]:
        start_time, end_time = session["times"][[item["start_frame"], item["end_frame"]]]
        start = min(int(np.searchsorted(times, start_time)), len(times) - 1)
        end = min(int(np.searchsorted(times, end_time)), len(times) - 1)
        intervals.append({**item, "start_frame": start, "end_frame": end,
                          "start_time": float(start_time), "end_time": float(end_time)})
    targets, mask = make_action_targets(valid, segments, intervals, session["reviewed"], config, excluded, times)
    return {**session, "times": times, "points": points, "world_points": world_points,
            "detected": detected, "features": features,
            "valid": valid, "segments": segments, "intervals": intervals, "targets": targets,
            "mask": mask, "ignored_frames": excluded, "available_times": np.asarray(available),
            "input_regions": input_regions,
            "source_session": session}


def scores_on_source_frames(session, predictions):
    source = session["source_session"]
    indices = np.searchsorted(session["available_times"], source["times"], side="right") - 1
    result = np.full(source["targets"].shape, np.nan, np.float32)
    usable = (indices >= 0) & source["valid"]
    if session.get("input_regions") is not None:
        usable &= source["input_regions"] == session["input_regions"][np.maximum(indices, 0)]
    result[usable] = predictions[indices[usable]]
    return source, result
