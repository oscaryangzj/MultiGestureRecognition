import argparse
import csv
import json

import numpy as np

from gesture.action_annotations import ActionAnnotationEditor
from gesture.action_data import load_action_session
from gesture.action_inference import action_output_directory
from gesture.action_ui import show_action_session
from gesture.config import load_config, project_path


def main():
    parser = argparse.ArgumentParser(description="Replay Action target pulses, masks and predicted scores")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--continuous", action="store_true", help="View the continuous replay comparison")
    args = parser.parse_args()
    config = load_config()
    session = load_action_session(config, args.session_id, isolate_negative=not args.continuous)
    output = action_output_directory(config) / "evaluation"
    folder = output / "continuous" / args.session_id if args.continuous else output / args.session_id
    path = folder / "predictions.csv"
    if not path.is_file():
        command = f"python -m scripts.evaluate_action --session-id {args.session_id}" + (" --continuous" if args.continuous else "")
        raise FileNotFoundError(f"Run {command} first")
    classes = config["labels"]["action_classes"]
    predictions = np.full_like(session["targets"], np.nan)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(session["times"]):
        raise ValueError("Prediction and video frame counts differ; reevaluate the session")
    for index, row in enumerate(rows):
        for channel, name in enumerate(classes):
            predictions[index, channel] = float(row[f"{name}_score"]) if row[f"{name}_score"] else np.nan
            session["targets"][index, channel] = float(row[f"{name}_target"])
            session["mask"][index, channel] = bool(int(row[f"{name}_supervised"]))
        session["valid"][index] = bool(int(row["valid"]))
    source = project_path(config["data"]["sessions_dir"]) / args.session_id
    annotations = json.loads((source / "annotations.json").read_text(encoding="utf-8"))
    editor = ActionAnnotationEditor(annotations, len(session["times"]), session["detected"], classes)
    show_action_session(config, session, editor, predictions, view_only=True)


if __name__ == "__main__":
    main()
