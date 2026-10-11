"""Application entry point — builds AppContext and launches GUI."""
import os
import sys
import logging
import traceback
import tempfile

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_aimusic_dir = os.path.dirname(os.path.abspath(__file__))

for _p in (_project_root, _aimusic_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import Config
from core import AppContext
from gui_main import launch_gui

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".aimusic")

def setup_logging():
    """Configure file and console logging."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    log_file = os.path.join(CONFIG_DIR, "aimusic_studio.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized → %s", log_file)


def main():
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting AI Music Studio...")
        logger.info("Python %s on %s", sys.version.split()[0], sys.platform)

        config = Config.load()
        ctx = AppContext(config)
        launch_gui(ctx)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user")
    except Exception as exc:
        logging.getLogger(__name__).critical(
            "Fatal error during startup:\n%s", traceback.format_exc()
        )
        raise


if __name__ == "__main__":
    main()