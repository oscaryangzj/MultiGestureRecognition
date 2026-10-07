import argparse
from urllib.request import urlretrieve

from gesture.config import load_config, project_path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


def main():
    parser = argparse.ArgumentParser(description="Download Google's Gesture Recognizer model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    destination = project_path(config["gesture_recognizer"]["model_asset_path"])
    if destination.exists() and not args.force:
        print(f"Model already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(MODEL_URL, destination)
    print(f"Saved official Gesture Recognizer model to {destination}")


if __name__ == "__main__":
    main()
