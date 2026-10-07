import json

from gesture.action_training import train_action
from gesture.config import load_config


def main():
    path, metrics = train_action(load_config())
    print(f"Saved Action model: {path}")
    print(json.dumps(metrics["validation"], indent=2))


if __name__ == "__main__":
    main()
