def frame_x(frame, first, last, left, right):
    """Map inclusive frame endpoints [first, last] to timeline endpoints."""
    if first == last:
        return left
    return left + round((frame - first) / (last - first) * (right - left))


def frame_at_x(x, first, last, left, right):
    """Select the nearest frame using the same coordinates as frame_x."""
    frame = first + round((x - left) / (right - left) * (last - first))
    return max(first, min(last, frame))
