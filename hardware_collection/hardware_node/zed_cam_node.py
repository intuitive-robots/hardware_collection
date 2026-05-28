"""ZED camera publisher that streams frames over ZeroLanCom."""

from __future__ import annotations

import argparse
from json import load
import logging
import signal
import time
from pathlib import Path
import threading
from typing import Any, Dict, List

import yaml

from hardware_collection.camera.camera_zed_sdk import ZED as ZEDCamera
import pyzlc



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
        pyzlc.error("Config file not found at %s, using CLI/defaults", cfg_path)
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping/dict")

    return data

def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
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

def _validate_camera_config(cfg: Dict[str, Any]) -> None:
    required_fields = [
        "device_id",
        "publish_topic",
        "width",
        "height",
        "fps",
        "depth_mode",
        "show_preview",
        "log_interval",
        "zlc_config",
    ]
    for field in required_fields:
        if field not in cfg or cfg[field] in (None, ""):
            raise ValueError(f"Missing required field '{field}' in the YAML configuration")
        

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    camera_cfgs = load_config(args.config)
    if _validate_camera_config(cfg):
        pyzlc.info("Camera configuration validated successfully.")
    pyzlc.info(f"Loaded camera configuration: {camera_cfgs}")
    camera = ZEDCamera(
                device_id=str(cfg["device_id"]),
                height=int(cfg["height"]),
                width=int(cfg["width"]),
                fps=int(cfg["fps"]),
                depth_mode=str(cfg["depth_mode"]),
                show_preview=bool(cfg["show_preview"]),
                publish_topic=cfg["publish_topic"],
                zlc_config=str(cfg["zlc_config"]),
                optional_settings_path=cfg.get("optional_settings_path"),
            )
    pyzlc.info(f"Initialized ZED camera with device name {cfg['publish_topic']}")
    pyzlc.info("[ZEDCamNode] Camera publisher started")
    frames_sent = 0
    last_report_time = time.time()
    log_interval = int(cfg["log_interval"])
    try:
        while True:
            camera.publish_frame()
            frames_sent += 1

            now = time.time()
            if now - last_report_time >= log_interval:
                elapsed = now - last_report_time
                fps = frames_sent / elapsed if elapsed > 0 else 0.0
                pyzlc.info("Published %d frames (%.2f FPS)", frames_sent, fps)
                frames_sent = 0
                last_report_time = now
    except Exception as exc:  # pragma: no cover - runtime feedback only
        pyzlc.error("Publisher stopped due to error: %s", exc)
        return 1
    finally:
        pyzlc.info("closing camera...")
        camera.close()
        time.sleep(1)  # Ensure all resources are cleaned up before exit
        pyzlc.info("ZED camera publisher exiting")
        return 0


if __name__ == "__main__":
    main()
