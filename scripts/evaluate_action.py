import argparse
import json

import torch

from gesture.action_data import action_split_ids, load_action_session
from gesture.action_decoding import (decode_action_session, events_on_source_frames,
                                     write_action_events, write_final_events)
from gesture.action_evaluation import action_event_metrics, action_failure_cases, aggregate_action_metrics
from gesture.action_inference import action_output_directory, load_action_model, predict_action_session, preprocessing_config, save_action_predictions
from gesture.config import load_config
from gesture.waving_fsm import trace_waving_session


def main():
    parser = argparse.ArgumentParser(description="Evaluate Action on complete sessions")
    parser.add_argument("--config", help="Use the config.yaml saved for a training run")
    parser.add_argument("--session-id", help="Evaluate one session; defaults to the Action validation split")
    parser.add_argument("--split", choices=("train", "val", "acceptance"), help="Evaluate a complete saved split")
    parser.add_argument("--continuous", action="store_true", help="Include preparation in input history as a continuous replay comparison")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    if torch.device(config["training"]["device"]).type == "cpu":
        torch.set_num_threads(int(config["training"]["action_cpu_threads"]))
    model, checkpoint = load_action_model(config)
    config = preprocessing_config(config, checkpoint)
    if args.session_id and args.split:
        parser.error("Use either --session-id or --split")
    split = args.split or "val"
    ids = [args.session_id] if args.session_id else action_split_ids(config, split)
    if not ids:
        raise ValueError(f"No Action sessions in the {split} split in splits.json")
    output = action_output_directory(config) / "evaluation"
    if args.continuous:
        output /= "continuous"
    elif not args.session_id:
        output /= split
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
        conflicts = []
        events = decode_action_session(session, scores, checkpoint["classes"], config, conflicts)
        trace = trace_waving_session(session, scores, checkpoint["classes"], config)
        write_action_events(folder / "events.csv", events_on_source_frames(session, events), conflicts)
        write_final_events(folder / "final_events.csv", events_on_source_frames(session, trace["final_events"]))
        save_action_predictions(folder / "predictions.csv", session, scores, checkpoint["classes"], trace)
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (folder / "failure_cases.json").write_text(
            json.dumps(action_failure_cases(session, metrics, config), indent=2), encoding="utf-8")
        print(f"{session_id}: {json.dumps(metrics, indent=2)}")
    summary = aggregate_action_metrics(evaluations)
    summary_path = output / (f"{args.session_id}_summary.json" if args.session_id else f"{split}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Complete-session summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
