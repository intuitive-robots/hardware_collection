"""Hand-eye calibration node for the DepthAI wrist camera.

This node subscribes to a wrist camera RGB stream and an end-effector pose stream,
detects a chessboard calibration target, and solves the eye-in-hand transform
with OpenCV's hand-eye calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import signal
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyzlc
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the wrist camera to the robot hand/end-effector."
    )
    parser.add_argument("--zlc-config", type=Path, default=Path("configs/zlc.yaml"))
    parser.add_argument("--node-name", type=str, default="wrist_cam_hand_eye_calibrator")
    parser.add_argument(
        "--wrist-topic",
        type=str,
        default="wrist_cam",
        help="DepthAI wrist camera frame topic. Defaults to configs/depthai_wrist.yaml device_name.",
    )
    parser.add_argument(
        "--pose-topic",
        type=str,
        default="robot/eef_pose",
        help=(
            "Topic publishing end-effector pose as a 4x4 matrix, position/quaternion, "
            "translation plus rotation matrix, or EE_pos/EE_quat."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/calibration/wrist_cam_hand_eye.yaml"))
    parser.add_argument(
        "--camera-info",
        type=Path,
        default=None,
        help="Optional YAML containing fx, fy, cx, cy and optional distortion coefficients.",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--dist-coeffs",
        type=float,
        nargs="*",
        default=None,
        help="Camera distortion coefficients. Defaults to zeros if omitted.",
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=9,
        help="Number of inner chessboard corners per row.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=6,
        help="Number of inner chessboard corners per column.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Chessboard square size in meters.",
    )
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--stabilize-delay",
        type=float,
        default=2.0,
        help="Wait this many seconds after first detecting the chessboard before accepting a sample.",
    )
    parser.add_argument(
        "--max-pose-age",
        type=float,
        default=0.25,
        help="Maximum timestamp difference between camera frame and pose sample in seconds.",
    )
    parser.add_argument(
        "--min-translation-delta",
        type=float,
        default=0.02,
        help="Minimum hand translation change from the last accepted sample in meters.",
    )
    parser.add_argument(
        "--min-rotation-delta-deg",
        type=float,
        default=5.0,
        help="Minimum hand rotation change from the last accepted sample in degrees.",
    )
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="tsai",
    )
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.pattern_cols <= 2 or args.pattern_rows <= 2:
        raise ValueError(
            "OpenCV chessboard detection requires both inner-corner dimensions to be > 2. "
            "For example, a board with 4x4 squares has 3x3 inner corners, so use "
            "--pattern-cols 3 --pattern-rows 3. A 3x3-squares board only has 2x2 "
            "inner corners and is too small for this calibration node."
        )
    return args


def load_zlc_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if "local_ip" not in config:
        raise ValueError(f"Missing 'local_ip' in {path}")
    return config


def msg_value(message: dict[Any, Any], key: str, default: Any = None) -> Any:
    if key in message:
        return message[key]
    return message.get(key.encode("utf-8"), default)


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


def as_float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def quaternion_to_rotation(quat: Any) -> np.ndarray:
    q = as_float_array(quat).reshape(4)
    x, y, z, w = q
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_transform(message: dict[Any, Any]) -> tuple[np.ndarray, float | None]:
    timestamp = msg_value(message, "timestamp", None)

    matrix = msg_value(message, "matrix", None)
    if matrix is None:
        matrix = msg_value(message, "transform", None)
    if matrix is None:
        matrix = msg_value(message, "pose_matrix", None)
    if matrix is not None:
        transform = as_float_array(matrix, (4, 4))
        return transform, None if timestamp is None else float(timestamp)

    position = msg_value(message, "position", None)
    if position is None:
        position = msg_value(message, "translation", None)
    if position is None:
        position = msg_value(message, "EE_pos", None)
    if position is None and all(msg_value(message, axis, None) is not None for axis in ("x", "y", "z")):
        position = [msg_value(message, "x"), msg_value(message, "y"), msg_value(message, "z")]

    rotation = msg_value(message, "rotation", None)
    if rotation is None:
        rotation = msg_value(message, "rotation_matrix", None)
    quaternion = msg_value(message, "quaternion", None)
    if quaternion is None:
        quaternion = msg_value(message, "EE_quat", None)
    if quaternion is None and all(
        msg_value(message, key, None) is not None for key in ("qx", "qy", "qz", "qw")
    ):
        quaternion = [
            msg_value(message, "qx"),
            msg_value(message, "qy"),
            msg_value(message, "qz"),
            msg_value(message, "qw"),
        ]

    if position is None:
        raise ValueError("Pose message is missing position/translation")
    if rotation is None and quaternion is None:
        raise ValueError("Pose message is missing rotation/quaternion")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = as_float_array(position, (3,))
    transform[:3, :3] = (
        as_float_array(rotation, (3, 3)) if rotation is not None else quaternion_to_rotation(quaternion)
    )
    return transform, None if timestamp is None else float(timestamp)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    cos_angle = (float(np.trace(delta)) - 1.0) * 0.5
    angle = math.acos(max(-1.0, min(1.0, cos_angle)))
    return math.degrees(angle)


def transform_to_dict(transform: np.ndarray) -> dict[str, Any]:
    return {
        "matrix": transform.tolist(),
        "translation": transform[:3, 3].tolist(),
        "rotation_matrix": transform[:3, :3].tolist(),
    }


def load_camera_model(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    camera_info: dict[str, Any] = {}
    if args.camera_info is not None:
        with args.camera_info.open("r", encoding="utf-8") as f:
            camera_info = yaml.safe_load(f) or {}

    fx = args.fx if args.fx is not None else camera_info.get("fx")
    fy = args.fy if args.fy is not None else camera_info.get("fy")
    cx = args.cx if args.cx is not None else camera_info.get("cx")
    cy = args.cy if args.cy is not None else camera_info.get("cy")
    missing = [name for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)) if value is None]
    if missing:
        raise ValueError(
            "Missing camera intrinsics: "
            + ", ".join(missing)
            + ". Provide --camera-info or --fx --fy --cx --cy."
        )

    dist_coeffs = args.dist_coeffs
    if dist_coeffs is None:
        dist_coeffs = camera_info.get("dist_coeffs", camera_info.get("distortion", [0, 0, 0, 0, 0]))

    camera_matrix = np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return camera_matrix, as_float_array(dist_coeffs).reshape(-1, 1)


class WristCamHandEyeCalibrator:
    def __init__(
        self,
        wrist_topic: str,
        pose_topic: str,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        pattern_size: tuple[int, int],
        square_size: float,
        max_queue_size: int,
    ) -> None:
        self.wrist_topic = wrist_topic
        self.pose_topic = pose_topic
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pattern_size = pattern_size
        self.object_points = self._make_object_points(pattern_size, square_size)
        self.frames: queue.Queue[dict[Any, Any]] = queue.Queue(maxsize=max_queue_size)
        self.latest_pose: tuple[np.ndarray, float | None] | None = None
        self.samples: list[dict[str, np.ndarray | float]] = []
        self.dropped_frames = 0
        self.running = True

    @staticmethod
    def _make_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
        cols, rows = pattern_size
        points = np.zeros((rows * cols, 3), dtype=np.float32)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        points *= float(square_size)
        return points

    def connect(self) -> None:
        pyzlc.register_subscriber_handler(self.wrist_topic, self._handle_frame)
        pyzlc.register_subscriber_handler(self.pose_topic, self._handle_pose)
        pyzlc.info("Subscribed to wrist camera topic '%s'", self.wrist_topic)
        pyzlc.info("Subscribed to end-effector pose topic '%s'", self.pose_topic)

    def _handle_frame(self, frame: dict[Any, Any]) -> None:
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def _handle_pose(self, pose: dict[Any, Any]) -> None:
        try:
            self.latest_pose = pose_to_transform(pose)
        except Exception as exc:  # pragma: no cover - runtime feedback only
            pyzlc.error("Ignoring invalid pose message: %s", exc)

    def pop_latest_frame(self) -> dict[Any, Any] | None:
        latest = None
        while True:
            try:
                latest = self.frames.get_nowait()
                self.frames.task_done()
            except queue.Empty:
                return latest

    def detect_target(self, frame: dict[Any, Any]) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray]:
        width = int(msg_value(frame, "width"))
        height = int(msg_value(frame, "height"))
        channels = int(msg_value(frame, "channels", 3))
        rgb_data = ensure_bytes(msg_value(frame, "rgb_data"))
        if rgb_data is None:
            raise ValueError("Camera frame is missing rgb_data")

        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((height, width, channels))
        if channels == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGBA2GRAY)
            preview = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            gray = rgb
            preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return False, None, None, preview

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            refined,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        cv2.drawChessboardCorners(preview, self.pattern_size, refined, ok)
        return bool(ok), rvec, tvec, preview

    def should_accept_sample(
        self,
        hand_to_base: np.ndarray,
        frame_timestamp: float | None,
        pose_timestamp: float | None,
        max_pose_age: float,
        min_translation_delta: float,
        min_rotation_delta_deg: float,
    ) -> tuple[bool, str]:
        if frame_timestamp is not None and pose_timestamp is not None:
            age = abs(frame_timestamp - pose_timestamp)
            if age > max_pose_age:
                return False, f"pose/frame timestamps differ by {age:.3f}s"

        if not self.samples:
            return True, "first sample"

        previous = self.samples[-1]["hand_to_base"]
        assert isinstance(previous, np.ndarray)
        translation_delta = float(np.linalg.norm(hand_to_base[:3, 3] - previous[:3, 3]))
        rot_delta = rotation_delta_deg(previous, hand_to_base)
        if translation_delta < min_translation_delta and rot_delta < min_rotation_delta_deg:
            return False, (
                f"motion too small: {translation_delta:.3f}m, {rot_delta:.1f}deg"
            )
        return True, f"motion: {translation_delta:.3f}m, {rot_delta:.1f}deg"

    def add_sample(
        self,
        hand_to_base: np.ndarray,
        target_rvec: np.ndarray,
        target_tvec: np.ndarray,
        frame_timestamp: float | None,
        pose_timestamp: float | None,
    ) -> None:
        target_rotation, _ = cv2.Rodrigues(target_rvec)
        self.samples.append(
            {
                "hand_to_base": hand_to_base.copy(),
                "target_to_camera_rotation": target_rotation,
                "target_to_camera_translation": target_tvec.reshape(3).astype(np.float64),
                "frame_timestamp": -1.0 if frame_timestamp is None else float(frame_timestamp),
                "pose_timestamp": -1.0 if pose_timestamp is None else float(pose_timestamp),
            }
        )

    def solve(self, method_name: str) -> tuple[np.ndarray, dict[str, float]]:
        method_map = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        r_gripper2base = [sample["hand_to_base"][:3, :3] for sample in self.samples]
        t_gripper2base = [sample["hand_to_base"][:3, 3] for sample in self.samples]
        r_target2cam = [sample["target_to_camera_rotation"] for sample in self.samples]
        t_target2cam = [sample["target_to_camera_translation"] for sample in self.samples]

        rotation, translation = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=method_map[method_name],
        )
        camera_to_hand = np.eye(4, dtype=np.float64)
        camera_to_hand[:3, :3] = rotation
        camera_to_hand[:3, 3] = translation.reshape(3)
        return camera_to_hand, self._estimate_residuals(camera_to_hand)

    def _estimate_residuals(self, camera_to_hand: np.ndarray) -> dict[str, float]:
        target_to_base_translations = []
        target_to_base_rotations = []
        for sample in self.samples:
            hand_to_base = sample["hand_to_base"]
            target_to_camera = np.eye(4, dtype=np.float64)
            target_to_camera[:3, :3] = sample["target_to_camera_rotation"]
            target_to_camera[:3, 3] = sample["target_to_camera_translation"]
            target_to_base = hand_to_base @ camera_to_hand @ target_to_camera
            target_to_base_translations.append(target_to_base[:3, 3])
            target_to_base_rotations.append(target_to_base)

        translations = np.asarray(target_to_base_translations)
        mean_translation = np.mean(translations, axis=0)
        translation_errors = np.linalg.norm(translations - mean_translation, axis=1)

        reference = target_to_base_rotations[0]
        rotation_errors = [rotation_delta_deg(reference, transform) for transform in target_to_base_rotations]
        return {
            "mean_target_translation_error_m": float(np.mean(translation_errors)),
            "max_target_translation_error_m": float(np.max(translation_errors)),
            "mean_target_rotation_error_deg": float(np.mean(rotation_errors)),
            "max_target_rotation_error_deg": float(np.max(rotation_errors)),
        }


def try_collect_stable_sample(
    calibrator: WristCamHandEyeCalibrator,
    args: argparse.Namespace,
    frame: dict[Any, Any],
    rvec: np.ndarray,
    tvec: np.ndarray,
    first_detected_time: float,
) -> tuple[bool, bool]:
    now = time.time()
    remaining = args.stabilize_delay - (now - first_detected_time)
    if remaining > 0:
        pyzlc.info("Chessboard detected, waiting %.2fs for camera to stabilize", remaining)
        return True, False

    hand_to_base, pose_timestamp = calibrator.latest_pose
    frame_timestamp_raw = msg_value(frame, "timestamp", None)
    frame_timestamp = None if frame_timestamp_raw is None else float(frame_timestamp_raw)
    accept, reason = calibrator.should_accept_sample(
        hand_to_base,
        frame_timestamp,
        pose_timestamp,
        args.max_pose_age,
        args.min_translation_delta,
        args.min_rotation_delta_deg,
    )
    if not accept:
        pyzlc.info("Skipping sample: %s", reason)
        return False, False

    calibrator.add_sample(hand_to_base, rvec, tvec, frame_timestamp, pose_timestamp)
    pyzlc.info("Accepted sample %d/%d (%s)", len(calibrator.samples), args.max_samples, reason)
    return False, True


def write_result(
    path: Path,
    args: argparse.Namespace,
    camera_to_hand: np.ndarray,
    residuals: dict[str, float],
    sample_count: int,
) -> None:
    hand_to_camera = invert_transform(camera_to_hand)
    data = {
        "camera": "wrist_cam",
        "calibration_type": "eye_in_hand",
        "sample_count": sample_count,
        "method": args.method,
        "pattern": {
            "type": "chessboard",
            "inner_corners": {"cols": args.pattern_cols, "rows": args.pattern_rows},
            "square_size_m": args.square_size,
        },
        "transforms": {
            "T_hand_camera": transform_to_dict(camera_to_hand),
            "T_camera_hand": transform_to_dict(hand_to_camera),
        },
        "residuals": residuals,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(data)), f, sort_keys=False)


def main() -> int:
    args = parse_args()
    zlc_config = load_zlc_config(args.zlc_config)
    pyzlc.init(
        args.node_name,
        zlc_config["local_ip"],
        group_name=zlc_config.get("group_name"),
        group_port=int(zlc_config.get("group_port", 7725)),
    )

    camera_matrix, dist_coeffs = load_camera_model(args)
    calibrator = WristCamHandEyeCalibrator(
        wrist_topic=args.wrist_topic,
        pose_topic=args.pose_topic,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size=args.square_size,
        max_queue_size=args.queue_size,
    )
    calibrator.connect()

    def request_stop(_signum: int, _frame: Any) -> None:
        calibrator.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pyzlc.info(
        "Collecting wrist_cam hand-eye samples. Need at least %d, will stop at %d.",
        args.min_samples,
        args.max_samples,
    )
    last_attempt_time = 0.0
    pending_detection_time: float | None = None

    while calibrator.running:
        frame = calibrator.pop_latest_frame()
        now = time.time()
        if frame is None or now - last_attempt_time < args.sample_interval:
            pyzlc.sleep(0.01)
            continue
        last_attempt_time = now

        if calibrator.latest_pose is None:
            pyzlc.info("Waiting for pose messages on '%s'", args.pose_topic)
            continue

        try:
            found, rvec, tvec, preview = calibrator.detect_target(frame)
        except Exception as exc:  # pragma: no cover - runtime feedback only
            pyzlc.error("Failed to process wrist camera frame: %s", exc)
            continue

        if args.preview:
            cv2.imshow("wrist_cam hand-eye calibration", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                calibrator.running = False

        if not found or rvec is None or tvec is None:
            pending_detection_time = None
            pyzlc.info("Chessboard not detected")
            continue

        if pending_detection_time is None:
            pending_detection_time = now

        pending, accepted = try_collect_stable_sample(
            calibrator,
            args,
            frame,
            rvec,
            tvec,
            pending_detection_time,
        )
        pending_detection_time = pending_detection_time if pending else None

        if accepted and len(calibrator.samples) >= args.max_samples:
            calibrator.running = False

    if args.preview:
        cv2.destroyAllWindows()

    if len(calibrator.samples) < args.min_samples:
        pyzlc.error(
            "Only collected %d samples; need at least %d.",
            len(calibrator.samples),
            args.min_samples,
        )
        return 1

    camera_to_hand, residuals = calibrator.solve(args.method)
    write_result(args.output, args, camera_to_hand, residuals, len(calibrator.samples))
    pyzlc.info("Wrote wrist camera hand-eye calibration to %s", args.output)
    pyzlc.info("Residuals: %s", residuals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
