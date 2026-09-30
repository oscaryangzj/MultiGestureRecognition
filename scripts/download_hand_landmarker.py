import argparse
from urllib.request import urlretrieve

from gesture.config import load_config, project_path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main():
    parser = argparse.ArgumentParser(description="Download Google's Hand Landmarker model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    destination = project_path(config["hand_landmarker"]["model_asset_path"])
    if destination.exists() and not args.force:
        print(f"Model already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(MODEL_URL, destination)
    print(f"Saved official Hand Landmarker model to {destination}")


if __name__ == "__main__":
    main()
