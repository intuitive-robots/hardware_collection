"""ZED camera publisher that streams frames over ZeroMQ."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from hardware_collection.camera.camera_zed_sdk import ZED as ZEDCamera


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream ZED frames over ZeroMQ")
    parser.add_argument("--device-id", required=True, help="Serial number of the ZED camera")
    parser.add_argument("--publish-port", type=int, default=6000, help="TCP port to publish frames")
    parser.add_argument(
        "--camera-port",
        type=int,
        default=7900,
        help="Internal port used when initializing the ZED SDK camera",
    )
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    parser.add_argument("--depth-mode", default="PERFORMANCE", help="ZED SDK depth mode")
    parser.add_argument("--show-preview", action="store_true", help="Display a preview window")
    parser.add_argument(
        "--log-interval", type=int, default=60, help="Seconds between FPS log messages"
    )
    return parser.parse_args()


class GracefulKiller:
    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *_):
        self.should_stop = True


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    publish_address = f"tcp://*:{args.publish_port}"
    logger.info("Starting ZED publisher on %s", publish_address)

    camera = ZEDCamera(
        device_id=args.device_id,
        port=args.camera_port,
        height=args.height,
        width=args.width,
        fps=args.fps,
        depth_mode=args.depth_mode,
        show_preview=args.show_preview,
        publish_address=publish_address,
    )

    killer = GracefulKiller()
    frames_sent = 0
    last_report_time = time.time()

    try:
        while not killer.should_stop:
            camera.publish_image()
            frames_sent += 1

            now = time.time()
            if now - last_report_time >= args.log_interval:
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
