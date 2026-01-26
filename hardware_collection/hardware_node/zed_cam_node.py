"""ZED camera publisher that streams frames over ZeroLanCom."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
import threading
from typing import Any, Dict

import yaml

from hardware_collection.camera.camera_zed_sdk import ZED as ZEDCamera
import pyzlc

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream ZED frames over ZeroLanCom")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/zed_publisher.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        logger.warning("Config file not found at %s, using CLI/defaults", cfg_path)
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping/dict")

    return data


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "publish_topic": None,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "depth_mode": "PERFORMANCE",
        "show_preview": False,
        "log_interval": 60,
    }

    file_cfg = load_config(args.config)
    merged = {**defaults, **file_cfg}

    if not merged.get("publish_topic"):
        raise ValueError("publish_topic must be provided in the YAML configuration")
    if "device_id" not in merged or not merged["device_id"]:
        raise ValueError("ZED device_id must be provided in the YAML configuration")

    return merged

def _Connect_cam():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load the camera configuration from the YAML file
    camera_config = load_config(args.config)

    # Validate required fields in the YAML configuration
    required_fields = ["device_id", "depth_mode", "publish_topic", "width", "height", "fps", "show_preview", "log_interval"]
    for field in required_fields:
        if field not in camera_config:
            raise ValueError(f"Missing required field '{field}' in the YAML configuration")

    pyzlc.init(camera_config["publish_topic"], "192.168.0.109")
    print(f"ZED Publisher initialized on topic: {camera_config['publish_topic']}")
    # Initialize the ZED camera with the configuration
    camera = ZEDCamera(
        device_id=str(camera_config["device_id"]),
        height=int(camera_config["height"]),
        width=int(camera_config["width"]),
        fps=int(camera_config["fps"]),
        depth_mode=str(camera_config["depth_mode"]),
        show_preview=bool(camera_config["show_preview"]),
        publish_topic=camera_config["publish_topic"],
    )

    frames_sent = 0
    last_report_time = time.time()
    log_interval = int(camera_config["log_interval"])
    
    try:
        while True:
            camera.publish_image()
            frames_sent += 1

            now = time.time()
            if now - last_report_time >= log_interval:
                elapsed = now - last_report_time
                fps = frames_sent / elapsed if elapsed > 0 else 0.0
                logger.info("Published %d frames (%.2f FPS)", frames_sent, fps)
                frames_sent = 0
                last_report_time = now
    except Exception as exc:  # pragma: no cover - runtime feedback only
        logger.error("Publisher stopped due to error: %s", exc)
        return 1
    finally:
        logger.info("close camera")
        camera.close()

if __name__ == "__main__":
    _Connect_cam()