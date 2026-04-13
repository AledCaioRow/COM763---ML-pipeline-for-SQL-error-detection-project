"""Compatibility entry point for synthetic SQL generation."""

from src.generate_queries import *  # noqa: F401,F403
from src.generate_queries import main


if __name__ == "__main__":
    main()

