import argparse
import json

from gesture.action_training import train_action
from gesture.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train the configured Action model")
    parser.add_argument("--config", help="Use a saved config.yaml for this training run")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    path, metrics = train_action(config)
    print(f"Saved Action model: {path}")
    print(json.dumps(metrics["validation"], indent=2))


if __name__ == "__main__":
    main()
