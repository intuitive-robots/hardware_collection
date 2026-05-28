"""Eye-to-hand calibration node for a fixed external camera.

This node subscribes to a fixed camera RGB stream and an end-effector pose stream.
It assumes the chessboard target is rigidly attached to the end-effector, then
estimates the fixed transform between robot base and camera.
"""

from __future__ import annotations

import argparse
from configparser import ConfigParser
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
import pyzlc
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from hardware_collection.hardware_node.wrist_cam_hand_eye_calibration_node import (
    WristCamHandEyeCalibrator,
    as_float_array,
    invert_transform,
    load_zlc_config,
    msg_value,
    pose_to_transform,
    quaternion_to_rotation,
    rotation_delta_deg,
    try_collect_stable_sample,
    transform_to_dict,
)


ZED_CALIBRATION_SECTIONS = {
    (2208, 1242): "LEFT_CAM_2K",
    (1920, 1080): "LEFT_CAM_FHD",
    (1280, 720): "LEFT_CAM_HD",
    (672, 376): "LEFT_CAM_VGA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a fixed external camera to the robot base. The chessboard "
            "target must be rigidly attached to the end-effector."
        )
    )
    parser.add_argument("--zlc-config", type=Path, default=Path("configs/zlc.yaml"))
    parser.add_argument("--node-name", type=str, default="eye_to_hand_calibrator")
    parser.add_argument(
        "--camera-topic",
        type=str,
        default="zed_depth",
        help="Fixed external camera frame topic. Defaults to the ZED depth camera topic.",
    )
    parser.add_argument(
        "--pose-topic",
        type=str,
        default="FrankaPanda/franka_arm_state",
        help=(
            "Topic publishing Panda arm state. Defaults to FrankaPanda/franka_arm_state "
            "and accepts O_T_EE or EE_pos/EE_quat."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data/calibration/eye_to_hand.yaml"))
    parser.add_argument(
        "--camera-info",
        type=Path,
        default=Path("/home/jjiang/.config/zed/settings/SN10905471.conf"),
        help="YAML containing fx, fy, cx, cy, or a ZED SN*.conf calibration file.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Camera image width used to select the ZED calibration section.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Camera image height used to select the ZED calibration section.",
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
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
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
        default=5.0,
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
            "Use inner corners, not square count. For a 10x7-squares board, use "
            "--pattern-cols 9 --pattern-rows 6."
        )
    return args


def load_zed_conf_camera_model(path: Path, width: int, height: int) -> dict[str, Any]:
    section = ZED_CALIBRATION_SECTIONS.get((int(width), int(height)))
    if section is None:
        supported = ", ".join(f"{w}x{h}" for w, h in ZED_CALIBRATION_SECTIONS)
        raise ValueError(
            f"No ZED calibration section mapping for {width}x{height}. "
            f"Supported resolutions: {supported}"
        )

    parser = ConfigParser()
    parser.optionxform = str.lower
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read ZED calibration file: {path}")
    if section not in parser:
        raise ValueError(f"Missing section [{section}] in {path}")

    cam = parser[section]
    return {
        "fx": float(cam["fx"]),
        "fy": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
        "dist_coeffs": [
            float(cam.get("k1", 0.0)),
            float(cam.get("k2", 0.0)),
            float(cam.get("k3", 0.0)),
            float(cam.get("k4", 0.0)),
        ],
        "calibration_file": str(path),
        "calibration_section": section,
    }


def load_camera_model(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    camera_info: dict[str, Any] = {}
    if args.camera_info is not None:
        if args.camera_info.suffix.lower() == ".conf":
            camera_info = load_zed_conf_camera_model(
                args.camera_info,
                args.camera_width,
                args.camera_height,
            )
        else:
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


def decode_panda_arm_state(message: Any) -> dict[Any, Any]:
    if isinstance(message, (bytes, bytearray)):
        return msgpack.unpackb(message, raw=False)
    if isinstance(message, dict):
        return message
    raise TypeError(f"Unsupported Panda arm state message type: {type(message).__name__}")


def comparable_epoch_timestamp(state: dict[Any, Any]) -> float | None:
    timestamp = msg_value(state, "timestamp", None)
    if timestamp is not None:
        timestamp_s = float(timestamp)
        if timestamp_s > 1e9:
            return timestamp_s

    time_ms = msg_value(state, "time_ms", None)
    if time_ms is not None:
        timestamp_s = float(time_ms) * 1e-3
        if timestamp_s > 1e9:
            return timestamp_s

    return None


def panda_arm_state_to_transform(message: Any) -> tuple[np.ndarray, float | None]:
    state = decode_panda_arm_state(message)
    timestamp = comparable_epoch_timestamp(state)

    o_t_ee = msg_value(state, "O_T_EE", None)
    if o_t_ee is not None:
        # Franka state matrices are conventionally flattened in column-major order.
        transform = as_float_array(o_t_ee, (4, 4)).T
        return transform, timestamp

    ee_pos = msg_value(state, "EE_pos", None)
    if ee_pos is None:
        return pose_to_transform(state)

    ee_quat = msg_value(state, "EE_quat", None)
    ee_rot = msg_value(state, "EE_rot", None)
    if ee_rot is None:
        ee_rot = msg_value(state, "O_R_EE", None)
    if ee_quat is None and ee_rot is None:
        raise ValueError(
            "Panda arm state has EE_pos but no orientation. Need O_T_EE, EE_quat, "
            "EE_rot, or O_R_EE for hand-eye calibration."
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = as_float_array(ee_pos, (3,))
    transform[:3, :3] = (
        as_float_array(ee_rot, (3, 3)) if ee_rot is not None else quaternion_to_rotation(ee_quat)
    )
    return transform, timestamp


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return transform


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    rotations = np.asarray([transform[:3, :3] for transform in transforms])
    translations = np.asarray([transform[:3, 3] for transform in transforms])

    rotation_mean = np.mean(rotations, axis=0)
    u, _s, vt = np.linalg.svd(rotation_mean)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    averaged = np.eye(4, dtype=np.float64)
    averaged[:3, :3] = rotation
    averaged[:3, 3] = np.mean(translations, axis=0)
    return averaged


def beep() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


class EyeToHandCalibrator(WristCamHandEyeCalibrator):
    def connect(self) -> None:
        pyzlc.register_subscriber_handler(self.wrist_topic, self._handle_frame)
        pyzlc.register_subscriber_handler(self.pose_topic, self._handle_pose)
        pyzlc.info("Subscribed to fixed camera topic '%s'", self.wrist_topic)
        pyzlc.info("Subscribed to end-effector pose topic '%s'", self.pose_topic)

    def _handle_pose(self, pose: Any) -> None:
        try:
            self.latest_pose = panda_arm_state_to_transform(pose)
        except Exception as exc:  # pragma: no cover - runtime feedback only
            pyzlc.error("Ignoring invalid Panda arm state message: %s", exc)

    def solve_eye_to_hand(self, method_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        method_map = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        r_gripper2base = [sample["hand_to_base"][:3, :3] for sample in self.samples]
        t_gripper2base = [sample["hand_to_base"][:3, 3] for sample in self.samples]

        target_to_camera = []
        camera_to_target = []
        for sample in self.samples:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = sample["target_to_camera_rotation"]
            transform[:3, 3] = sample["target_to_camera_translation"]
            target_to_camera.append(transform)
            camera_to_target.append(invert_transform(transform))

        r_camera2target = [transform[:3, :3] for transform in camera_to_target]
        t_camera2target = [transform[:3, 3] for transform in camera_to_target]

        hand_target_rotation, hand_target_translation = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_camera2target,
            t_camera2target,
            method=method_map[method_name],
        )
        hand_to_target = make_transform(hand_target_rotation, hand_target_translation)

        base_to_camera_estimates = []
        for sample, camera_to_target_transform in zip(self.samples, camera_to_target):
            hand_to_base = sample["hand_to_base"]
            base_to_camera_estimates.append(hand_to_base @ hand_to_target @ camera_to_target_transform)

        base_to_camera = average_transforms(base_to_camera_estimates)
        return base_to_camera, hand_to_target, self._estimate_eye_to_hand_residuals(
            base_to_camera,
            hand_to_target,
            camera_to_target,
        )

    def _estimate_eye_to_hand_residuals(
        self,
        base_to_camera: np.ndarray,
        hand_to_target: np.ndarray,
        camera_to_target: list[np.ndarray],
    ) -> dict[str, float]:
        estimates = []
        for sample, camera_to_target_transform in zip(self.samples, camera_to_target):
            hand_to_base = sample["hand_to_base"]
            estimates.append(hand_to_base @ hand_to_target @ camera_to_target_transform)

        translations = np.asarray([transform[:3, 3] for transform in estimates])
        translation_errors = np.linalg.norm(translations - base_to_camera[:3, 3], axis=1)
        rotation_errors = [rotation_delta_deg(base_to_camera, transform) for transform in estimates]

        return {
            "mean_base_camera_translation_error_m": float(np.mean(translation_errors)),
            "max_base_camera_translation_error_m": float(np.max(translation_errors)),
            "mean_base_camera_rotation_error_deg": float(np.mean(rotation_errors)),
            "max_base_camera_rotation_error_deg": float(np.max(rotation_errors)),
        }


def write_result(
    path: Path,
    args: argparse.Namespace,
    base_to_camera: np.ndarray,
    hand_to_target: np.ndarray,
    residuals: dict[str, float],
    sample_count: int,
) -> None:
    camera_to_base = invert_transform(base_to_camera)
    target_to_hand = invert_transform(hand_to_target)
    data = {
        "camera": args.camera_topic,
        "calibration_type": "eye_to_hand",
        "sample_count": sample_count,
        "method": args.method,
        "assumption": "fixed camera, chessboard target rigidly attached to end-effector",
        "pattern": {
            "type": "chessboard",
            "inner_corners": {"cols": args.pattern_cols, "rows": args.pattern_rows},
            "square_size_m": args.square_size,
        },
        "transforms": {
            "T_base_camera": transform_to_dict(base_to_camera),
            "T_camera_base": transform_to_dict(camera_to_base),
            "T_hand_target": transform_to_dict(hand_to_target),
            "T_target_hand": transform_to_dict(target_to_hand),
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
    calibrator = EyeToHandCalibrator(
        wrist_topic=args.camera_topic,
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
        "Collecting eye-to-hand samples. Need at least %d, will stop at %d.",
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
            pyzlc.error("Failed to process camera frame: %s", exc)
            continue

        if args.preview:
            cv2.imshow("eye-to-hand calibration", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                calibrator.running = False

        if not found or rvec is None or tvec is None:
            pending_detection_time = None
            pyzlc.info("Chessboard not detected")
            continue
        beep()

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

    base_to_camera, hand_to_target, residuals = calibrator.solve_eye_to_hand(args.method)
    write_result(args.output, args, base_to_camera, hand_to_target, residuals, len(calibrator.samples))
    pyzlc.info("Wrote eye-to-hand calibration to %s", args.output)
    pyzlc.info("Residuals: %s", residuals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
