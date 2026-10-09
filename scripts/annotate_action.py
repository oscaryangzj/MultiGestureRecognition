import argparse

from gesture.action_annotations import ActionAnnotationEditor
from gesture.action_data import load_action_session
from gesture.action_ui import show_action_session
from gesture.config import load_config
from gesture.periodic_annotations import PeriodicAnnotationEditor


def main():
    parser = argparse.ArgumentParser(description="Review positive events or negative tasks one prompt at a time")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    config = load_config()
    session = load_action_session(config, args.session_id)
    if not session["opportunities"] and session.get("kind") != "continuous":
        raise ValueError("This session has no recorded Action prompts to review")
    annotations = session["annotations"]
    editor_class = PeriodicAnnotationEditor if session.get("kind") == "periodic" else ActionAnnotationEditor
    editor = editor_class(annotations, len(session["times"]), session["detected"], config["labels"]["action_classes"], session, config)
    show_action_session(config, session, editor)


if __name__ == "__main__":
    main()
