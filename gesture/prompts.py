import numpy as np


REST_LABEL = "REST"
HOLD_LABEL = "HOLD"


def prompt_labels_for_session(session, frame_count):
    """Load recorded task ranges and preparation/hold gaps."""
    gap_label = "PREPARE" if session.get("sample_type") == "negative" else HOLD_LABEL if session.get("pattern") == "alternating" else REST_LABEL
    labels = np.full(frame_count, gap_label, dtype=object)
    for prompt in session["prompts"]:
        if prompt.get("start_frame") is None or prompt.get("end_frame") is None:
            continue
        start = int(prompt["start_frame"])
        end = int(prompt["end_frame"])
        if start < 0 or end < start or end >= frame_count:
            raise ValueError(f"Invalid recorded prompt range: {prompt}")
        labels[start : end + 1] = "NEGATIVE" if prompt.get("sample_type") == "negative" else prompt["gesture"]
    return labels


def label_segments(labels):
    """Yield inclusive ranges of consecutive equal labels."""
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            yield labels[start], start, end - 1
            start = end
