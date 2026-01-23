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


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream ZED frames over ZeroLanCom")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/zed_publisher.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--zed-config-dir",
        type=str,
        default=None,
        help="Directory containing per-camera ZED config YAMLs",
    )
    parser.add_argument(
        "--global-config",
        type=str,
        default="configs/config.yaml",
        help="Path to global config with camera_streams topics",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        help="Camera entry name in global config (e.g., zed_left)",
    )
    parser.add_argument("--device-id", help="Serial number of the ZED camera")
    parser.add_argument("--publish-topic", help="ZeroLanCom topic to publish frames to")
    parser.add_argument("--width", type=int, help="Frame width")
    parser.add_argument("--height", type=int, help="Frame height")
    parser.add_argument("--fps", type=int, help="Target capture FPS")
    parser.add_argument("--depth-mode", help="ZED SDK depth mode")
    parser.add_argument(
        "--show-preview",
        dest="show_preview",
        action="store_true",
        help="Display a preview window",
    )
    parser.add_argument(
        "--no-preview",
        dest="show_preview",
        action="store_false",
        help="Disable preview window",
    )
    parser.set_defaults(show_preview=None)
    parser.add_argument(
        "--log-interval", type=int, help="Seconds between FPS log messages"
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


def _load_topic_from_global(config_path: str, camera_name: str | None = None) -> str | None:
    """Read camera_streams from a global config and return the selected topic."""
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        logger.warning("Global config %s not found; please supply --publish-topic explicitly", cfg_path)
        return None

    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to read global config %s: %s", cfg_path, exc)
        return None

    streams = data.get("camera_streams", {})
    if not isinstance(streams, dict):
        return None

    if camera_name:
        stream = streams.get(camera_name)
        if isinstance(stream, dict):
            topic = stream.get("topic")
            if topic:
                return topic
        logger.warning("Camera '%s' not found or missing topic in %s", camera_name, cfg_path)

    # fallback: first stream whose name starts with "zed"
    for name, stream in streams.items():
        if not isinstance(stream, dict):
            continue
        topic = stream.get("topic")
        if topic and name.startswith("zed"):
            return topic

    # final fallback: first available topic
    for stream in streams.values():
        if isinstance(stream, dict) and stream.get("topic"):
            return stream["topic"]

    logger.warning("No camera topics found in %s", cfg_path)
    return None


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "publish_topic": None,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "depth_mode": "PERFORMANCE",
        "show_preview": False,
        "log_interval": 60,
        "camera_name": None,
        "global_config": "configs/config.yaml",
        "camera_configs": None,
        "zed_config_dir": "hardware_collection/zed_config",
    }

    file_cfg = load_config(args.config)
    merged = {**defaults, **file_cfg}

    for key in defaults.keys() | {"device_id"}:
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val

    if getattr(args, "zed_config_dir", None):
        merged["zed_config_dir"] = args.zed_config_dir

    if not merged.get("publish_topic"):
        merged["publish_topic"] = _load_topic_from_global(
            merged.get("global_config", "configs/config.yaml"),
            camera_name=merged.get("camera_name"),
        )

    if not merged.get("camera_configs"):
        if "device_id" not in merged or merged["device_id"] in (None, ""):
            raise ValueError("ZED device_id must be provided (config or CLI)")
        if not merged.get("publish_topic"):
            raise ValueError("publish_topic not provided and not found in global config")

    return merged


class GracefulKiller:
    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *_):
        self.should_stop = True


def _load_camera_config_files(camera_configs: list[str], base_dir: str) -> list[dict]:
    cfgs: list[dict] = []
    base_path = Path(base_dir)
    for item in camera_configs:
        cfg_path = Path(item)
        if not cfg_path.is_absolute():
            cfg_path = base_path / cfg_path
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Camera config not found: {cfg_path}")
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Camera config at {cfg_path} must be a mapping/dict")
        data["_config_path"] = str(cfg_path)
        cfgs.append(data)
    return cfgs


def _merge_camera_defaults(cam_cfg: dict, defaults: dict) -> dict:
    merged = {**defaults, **cam_cfg}
    if not merged.get("device_id"):
        raise ValueError("device_id missing in camera config")
    if not merged.get("publish_topic"):
        raise ValueError("publish_topic missing in camera config")
    return merged


def _camera_loop(
    camera: ZEDCamera,
    log_interval: int,
    stop_event: threading.Event,
    logger_name: str,
) -> None:
    local_logger = logging.getLogger(logger_name)
    frames_sent = 0
    last_report_time = time.time()
    try:
        while not stop_event.is_set():
            camera.publish_image()
            frames_sent += 1

            now = time.time()
            if now - last_report_time >= log_interval:
                elapsed = now - last_report_time
                fps = frames_sent / elapsed if elapsed > 0 else 0.0
                local_logger.info("Published %d frames (%.2f FPS)", frames_sent, fps)
                frames_sent = 0
                last_report_time = now
    except Exception as exc:  # pragma: no cover - runtime feedback only
        local_logger.error("Publisher stopped due to error: %s", exc)
    finally:
        camera.close()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = resolve_config(args)

    killer = GracefulKiller()
    if cfg.get("camera_configs"):
        defaults = {
            "width": cfg["width"],
            "height": cfg["height"],
            "fps": cfg["fps"],
            "depth_mode": cfg["depth_mode"],
            "show_preview": cfg["show_preview"],
            "log_interval": cfg["log_interval"],
        }
        camera_cfgs = _load_camera_config_files(
            cfg["camera_configs"],
            cfg.get("zed_config_dir", "hardware_collection/zed_config"),
        )
        cameras: list[ZEDCamera] = []
        threads: list[threading.Thread] = []
        stop_event = threading.Event()

        logger.info("Starting %d ZED publishers from config list", len(camera_cfgs))
        for idx, cam_cfg in enumerate(camera_cfgs):
            merged = _merge_camera_defaults(cam_cfg, defaults)
            publish_topic = merged["publish_topic"]
            config_path = merged.get("_config_path", "unknown")
            logger.info(
                "Camera config %d: %s | topic=%s | serial=%s",
                idx,
                config_path,
                publish_topic,
                merged.get("device_id"),
            )
            logger.info(
                "Starting ZED publisher %d on ZeroLanCom topic '%s'", idx, publish_topic
            )
            camera = ZEDCamera(
                device_id=str(merged["device_id"]),
                height=int(merged["height"]),
                width=int(merged["width"]),
                fps=int(merged["fps"]),
                depth_mode=str(merged["depth_mode"]),
                show_preview=bool(merged["show_preview"]),
                publish_topic=publish_topic,
            )
            cameras.append(camera)
            thread = threading.Thread(
                target=_camera_loop,
                args=(
                    camera,
                    int(merged["log_interval"]),
                    stop_event,
                    f"{__name__}.cam{idx}",
                ),
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        try:
            while not killer.should_stop:
                time.sleep(0.1)
        finally:
            logger.info("Shutting down publishers")
            stop_event.set()
            for thread in threads:
                thread.join(timeout=5.0)
    else:
        publish_topic = cfg["publish_topic"]
        logger.info("Starting ZED publisher on ZeroLanCom topic '%s'", publish_topic)

        camera = ZEDCamera(
            device_id=str(cfg["device_id"]),
            height=cfg["height"],
            width=cfg["width"],
            fps=cfg["fps"],
            depth_mode=str(cfg["depth_mode"]),
            show_preview=bool(cfg["show_preview"]),
            publish_topic=publish_topic,
        )

        frames_sent = 0
        last_report_time = time.time()
        log_interval = cfg["log_interval"]

        try:
            while not killer.should_stop:
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
            logger.info("Shutting down publisher")
            camera.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
