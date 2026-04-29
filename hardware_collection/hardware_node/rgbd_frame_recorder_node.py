"""Subscribe to static/wrist camera RGB-D streams and save received frames."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyzlc
import yaml


DEFAULT_CAMERA_TOPICS = {
    "static_cam": "static_cam",
    "wrist_cam": "wrist_cam",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to static/wrist RGB-D camera topics and save frames."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rgbd_frames"),
        help="Directory where frames will be saved.",
    )
    parser.add_argument(
        "--zlc-config",
        type=Path,
        default=Path("configs/zlc.yaml"),
        help="Path to ZeroLanCom YAML config.",
    )
    parser.add_argument(
        "--node-name",
        type=str,
        default="rgbd_frame_recorder",
        help="ZeroLanCom node name for this subscriber.",
    )
    parser.add_argument(
        "--static-topic",
        type=str,
        default=DEFAULT_CAMERA_TOPICS["static_cam"],
        help="Topic published by the static camera node.",
    )
    parser.add_argument(
        "--wrist-topic",
        type=str,
        default=DEFAULT_CAMERA_TOPICS["wrist_cam"],
        help="Topic published by the wrist camera node.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after saving this many frames per camera. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=1024,
        help="Maximum pending frame messages before dropping new frames.",
    )
    return parser.parse_args()


def load_zlc_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if "local_ip" not in config:
        raise ValueError(f"Missing 'local_ip' in {path}")
    return config


def frame_value(frame: dict[Any, Any], key: str, default: Any = None) -> Any:
    if key in frame:
        return frame[key]
    return frame.get(key.encode("utf-8"), default)


def ensure_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


class RGBDFrameRecorder:
    def __init__(
        self,
        output_dir: Path,
        camera_topics: dict[str, str],
        max_queue_size: int,
    ) -> None:
        self.output_dir = output_dir / time.strftime("%Y_%m_%d-%H_%M_%S")
        self.camera_topics = camera_topics
        self.frames: queue.Queue[tuple[str, dict[Any, Any]]] = queue.Queue(maxsize=max_queue_size)
        self.frame_counts = {name: 0 for name in camera_topics}
        self.drop_counts = {name: 0 for name in camera_topics}
        self.running = True

        for camera_name in camera_topics:
            for subdir in ("rgb", "depth", "metadata"):
                (self.output_dir / camera_name / subdir).mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        for camera_name, topic in self.camera_topics.items():
            pyzlc.register_subscriber_handler(topic, self._make_handler(camera_name))
            pyzlc.info("Subscribed to %s on topic '%s'", camera_name, topic)

    def _make_handler(self, camera_name: str):
        def handle_frame(frame: dict[Any, Any]) -> None:
            try:
                self.frames.put_nowait((camera_name, frame))
            except queue.Full:
                self.drop_counts[camera_name] += 1

        return handle_frame

    def save_pending_frames(self) -> int:
        saved = 0
        while True:
            try:
                camera_name, frame = self.frames.get_nowait()
            except queue.Empty:
                break

            try:
                self.save_frame(camera_name, frame)
                saved += 1
            except Exception as exc:  # pragma: no cover - runtime feedback only
                pyzlc.error("Failed to save %s frame: %s", camera_name, exc)
            finally:
                self.frames.task_done()

        return saved

    def save_frame(self, camera_name: str, frame: dict[Any, Any]) -> None:
        width = int(frame_value(frame, "width"))
        height = int(frame_value(frame, "height"))
        channels = int(frame_value(frame, "channels", 3))
        timestamp = frame_value(frame, "timestamp", time.time())
        rgb_data = ensure_bytes(frame_value(frame, "rgb_data"))
        depth_data = ensure_bytes(frame_value(frame, "depth_data"))

        if rgb_data is None:
            raise ValueError("Frame is missing rgb_data")

        frame_idx = self.frame_counts[camera_name]
        self.frame_counts[camera_name] += 1

        camera_dir = self.output_dir / camera_name
        stem = f"{frame_idx:06d}"

        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((height, width, channels))
        if channels == 3:
            rgb_to_save = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            rgb_to_save = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA)
        else:
            rgb_to_save = rgb
        rgb_path = camera_dir / "rgb" / f"{stem}.png"
        cv2.imwrite(str(rgb_path), rgb_to_save)

        depth_path = None
        if depth_data:
            depth_path = self._save_depth(camera_dir, stem, depth_data)

        metadata = {
            "camera_name": camera_name,
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "channels": channels,
            "rgb_path": str(rgb_path.relative_to(self.output_dir)),
            "depth_path": None if depth_path is None else str(depth_path.relative_to(self.output_dir)),
            "depth_format": "raw",
            "depth_bytes": 0 if depth_data is None else len(depth_data),
        }
        metadata_path = camera_dir / "metadata" / f"{stem}.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _save_depth(
        self,
        camera_dir: Path,
        stem: str,
        depth_data: bytes,
    ) -> Path:
        depth_path = camera_dir / "depth" / f"{stem}.raw"
        depth_path.write_bytes(depth_data)
        return depth_path

    def reached_max_frames(self, max_frames: int) -> bool:
        if max_frames <= 0:
            return False
        return all(count >= max_frames for count in self.frame_counts.values())


def main() -> int:
    args = parse_args()
    zlc_config = load_zlc_config(args.zlc_config)
    pyzlc.init(
        args.node_name,
        zlc_config["local_ip"],
        group_name=zlc_config.get("group_name"),
        group_port=int(zlc_config.get("group_port", 7725)),
    )

    recorder = RGBDFrameRecorder(
        output_dir=args.output_dir,
        camera_topics={
            "static_cam": args.static_topic,
            "wrist_cam": args.wrist_topic,
        },
        max_queue_size=args.queue_size,
    )
    recorder.connect()

    def request_stop(_signum: int, _frame: Any) -> None:
        recorder.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pyzlc.info("Saving RGB-D frames under %s", recorder.output_dir)
    start_time = time.time()
    last_report_time = start_time

    while recorder.running:
        recorder.save_pending_frames()
        now = time.time()

        if args.duration > 0 and now - start_time >= args.duration:
            recorder.running = False
        if recorder.reached_max_frames(args.max_frames):
            recorder.running = False

        if now - last_report_time >= 5.0:
            pyzlc.info(
                "Saved frames: %s, dropped frames: %s",
                recorder.frame_counts,
                recorder.drop_counts,
            )
            last_report_time = now

        pyzlc.sleep(0.01)

    recorder.save_pending_frames()
    pyzlc.info("Final saved frames: %s", recorder.frame_counts)
    pyzlc.info("Dropped frames: %s", recorder.drop_counts)
    pyzlc.info("RGB-D recorder exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
