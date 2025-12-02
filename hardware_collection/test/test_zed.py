"""
Basic sanity test for hardware_collection.camera.camera_zed_sdk.ZED.
- Opens the camera
- Captures a single frame
- Prints header info and checks depth retrieval

Usage:
    python test_zed.py --device-id 30414018 --port 7900 --width 1280 --height 720 --fps 30 --depth-mode PERFORMANCE

Note: Requires ZED SDK + pyzed installed and an actual camera connected.
"""
import argparse
import sys
import time
import logging

from hardware_collection.camera.camera_zed_sdk import ZED

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("test_zed")

#python test_zed.py --device-id 30414018 --port 7900 --width 1280 --height 720 --fps 30 --depth-mode PERFORMANCE --show-preview 
def parse_args():
    parser = argparse.ArgumentParser(description="Basic ZED SDK camera sanity test")
    parser.add_argument("--device-id", required=True, help="Camera serial number")
    parser.add_argument("--port", type=int, default=7900, help="Internal port used by SDK instance")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument(
        "--depth-mode",
        default="PERFORMANCE",
        choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"],
        help="ZED depth mode",
    )
    parser.add_argument("--show-preview", action="store_true", help="Show preview window")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Connecting to ZED (serial=%s)", args.device_id)
    cam = ZED(
        device_id=args.device_id,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        depth_mode=args.depth_mode,
        show_preview=args.show_preview,
    )

    try:
        logger.info("Capturing a frame...")
        frame = cam.capture_image()
        logger.info(
            "Captured frame id=%s size=%sx%s channels=%s ts=%.3f",
            frame.header.frame_id,
            frame.header.width,
            frame.header.height,
            frame.header.channels,
            frame.header.timestamp,
        )

        # Depth sanity check
        depth = cam.get_latest_depth()
        if depth is None:
            logger.warning("Depth data is None")
        else:
            logger.info("Depth shape=%s dtype=%s nan_count=%d",
                        depth.shape, depth.dtype, int((depth != depth).sum()))

        # Camera info
        info = cam.get_camera_information()
        logger.info("Camera info: %s", info)

        # Simple FPS measurement: capture 10 frames
        logger.info("Measuring short FPS over 10 frames...")
        start = time.time()
        for _ in range(10):
            cam.capture_image()
        elapsed = time.time() - start
        fps = 10 / elapsed if elapsed > 0 else 0.0
        logger.info("Approx FPS over 10 frames: %.2f", fps)

    except Exception as exc:
        logger.error("Test failed: %s", exc)
        return 1
    finally:
        cam.close()

    logger.info("Test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
