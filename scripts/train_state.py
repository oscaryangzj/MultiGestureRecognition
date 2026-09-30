import json

from gesture.config import load_config
from gesture.training import train_state


def main():
    checkpoint_path, metrics = train_state(load_config())
    print(f"Saved State model: {checkpoint_path}")
    print(json.dumps(metrics["validation"], indent=2))


if __name__ == "__main__":
    main()
