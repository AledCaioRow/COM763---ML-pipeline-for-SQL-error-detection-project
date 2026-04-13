"""Compatibility entry point for model training."""

from src.train import *  # noqa: F401,F403
from src.train import main


if __name__ == "__main__":
    main()

