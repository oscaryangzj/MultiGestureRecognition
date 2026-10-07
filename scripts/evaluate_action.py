import argparse
import json

from gesture.action_data import action_split_ids, load_action_session
from gesture.action_evaluation import action_event_metrics, aggregate_action_metrics
from gesture.action_inference import action_output_directory, load_action_model, predict_action_session, preprocessing_config, save_action_predictions
from gesture.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Action on complete sessions")
    parser.add_argument("--session-id", help="Evaluate one session; defaults to the Action validation split")
    parser.add_argument("--continuous", action="store_true", help="Include preparation in input history as a continuous replay comparison")
    args = parser.parse_args()
    config = load_config()
    model, checkpoint = load_action_model(config)
    config = preprocessing_config(config, checkpoint)
    ids = [args.session_id] if args.session_id else action_split_ids(config, "val")
    if not ids:
        raise ValueError("No Action validation sessions in splits.json")
    output = action_output_directory(config) / "evaluation"
    if args.continuous:
        output /= "continuous"
    evaluations = {}
    for session_id in ids:
        session = load_action_session(config, session_id, resample="sample_rate_hz" in checkpoint,
                                      isolate_negative=not args.continuous)
        if not session["reviewed"]:
            raise ValueError(f"{session_id}: review or cancel every Action/background prompt before evaluation")
        scores = predict_action_session(model, session, checkpoint, int(config["training"]["action_batch_size"]))
        metrics = action_event_metrics(session, scores, config)
        evaluations[session_id] = metrics
        folder = output / session_id
        save_action_predictions(folder / "predictions.csv", session, scores, checkpoint["classes"])
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"{session_id}: {json.dumps(metrics, indent=2)}")
        print(f"View: python -m scripts.view_action --session-id {session_id}" + (" --continuous" if args.continuous else ""))
    summary = aggregate_action_metrics(evaluations)
    summary_path = output / ("validation_summary.json" if not args.session_id else f"{args.session_id}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Complete-session summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
