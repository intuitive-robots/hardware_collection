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
import cv2
import numpy as np

from numpy import diag_indices
import yaml

from hardware_collection.camera.camera_depthai import DepthAICamera as DAIcam
import pyzlc

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream DepthAI frames over ZeroLanCom")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/depthai_publisher.yaml",
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
        "device_id": None,
        "publish_topic": None,
        "width": 512,
        "height": 512,
        "camera_type": "OAK_D_LITE",
        "log_interval": 60
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
    required_fields = ["device_id", "publish_topic", "width", "height", "camera_type", "log_interval"]
    for field in required_fields:
        if field not in camera_config:
            raise ValueError(f"Missing required field '{field}' in the YAML configuration")

    # Extract camera_type name if it's a dict
    camera_type = camera_config.get("camera_type")
    if isinstance(camera_type, dict) and "name" in camera_type:
        camera_config["camera_type"] = camera_type["name"]

    # Convert to DAICameraType enum
    from hardware_collection.camera.camera_depthai import DAICameraType
    camera_type_enum = DAICameraType[camera_config["camera_type"]]

    pyzlc.init(camera_config["publish_topic"], "192.168.0.109")
    print(f"ZED Publisher initialized on topic: {camera_config['publish_topic']}")

    # Initialize the DepthAI camera with the configuration
    camera = DAIcam(
        device_id=str(camera_config["device_id"]),
        height=int(camera_config["height"]),
        width=int(camera_config["width"]),
        camera_type=camera_type_enum,
        publish_topic=camera_config["publish_topic"],
    )

    frames_sent = 0
    last_report_time = time.time()
    log_interval = int(camera_config.get("log_interval", 60))
    
    try:

        frame = camera.capture_image()
        # If capture_image returns a CameraFrame, extract the image_data
        if hasattr(frame, 'image_data'):
            img = frame.image_data
        else:
            img = frame
        if img is not None and isinstance(img, np.ndarray):
                out_path = "/home/multimodallearning/Pictures/depthai_single_frame.jpg"
                cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved single frame to {out_path}")
        else:
            print("Warning: Invalid frame received!")
        while True:
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