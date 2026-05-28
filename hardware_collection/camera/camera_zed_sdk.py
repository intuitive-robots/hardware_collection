from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from .camera import AbstractCamera, CameraFrame


logger = logging.getLogger(__name__)


class ZED(AbstractCamera):
    """ZED camera that streams RGB frames over ZeroLanCom using the ZED SDK."""

    _RESOLUTION_MAP = {
        (2208, 1242): "HD2K",
        (1920, 1080): "HD1080",
        (1280, 720): "HD720",
        (672, 376): "VGA",
    }

    _DEPTH_MODES = ("NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL")

    def __init__(
        self,
        device_id: str,
        *,  # use keyword-only arguments after this
        device_name: Optional[str] = None,
        publish_topic: Optional[str] = None,
        name: Optional[str] = None,
        height: int = 720,
        width: int = 1280,
        fps: int = 30,
        depth_mode: str = "PERFORMANCE",
        show_preview: bool = False,
        zlc_config: Optional[str] = None,
        optional_settings_path: Optional[str] = None,
    ) -> None:
        self.device_id = str(device_id)
        self.name = name or f"ZED_{self.device_id}"
        self.fps = fps
        self.depth_mode = depth_mode.upper()
        self.optional_settings_path = optional_settings_path
        self.depth_enabled = self.depth_mode != "NONE"
        self._sl = None
        self.zed = None
        self.runtime_parameters = None
        self.image = None
        self.depth = None
        self.latest_depth: Optional[np.ndarray] = None

        resolved_device_name = device_name or publish_topic or f"zed_{self.device_id}"
        if not zlc_config:
            raise ValueError("zlc_config is required to initialize ZeroLanCom.")

        super().__init__(
            device_name=resolved_device_name,
            width=width,
            height=height,
            show_preview=show_preview,
            depth=self.depth_enabled,
            zlc_config=zlc_config,
        )
        self.initialize()

    @staticmethod
    def _load_sdk():
        try:
            import pyzed.sl as sl
        except (ImportError, AttributeError):
            try:
                import pyzed as sl
            except ImportError as exc:
                raise ImportError(
                    "pyzed package not found. Please install the ZED SDK and pyzed package."
                ) from exc
        return sl

    def initialize(self) -> None:
        """Initialize connections to the ZED camera via the SDK."""
        self._sl = self._load_sdk()
        self.zed = self._sl.Camera()
        init_params = self._sl.InitParameters()
        # init_params.camera_serial_number = int(self.device_id)
        input_type = self._sl.InputType()
        input_type.set_from_serial_number(int(self.device_id))
        init_params.open_timeout_sec = 60.0  # Gibt dem USB-Bus Zeit (WICHTIG!)
        init_params.sdk_verbose = 1          # Zeigt Logs im Terminal, falls es hängt
        init_params.async_grab_camera_recovery = True # Versucht Frame-Recovery bei USB-Hickups

        init_params.input = input_type
        init_params.camera_fps = self.fps
        init_params.depth_mode = self._get_depth_mode()
        init_params.coordinate_units = self._sl.UNIT.MILLIMETER
        init_params.camera_resolution = self._select_resolution(init_params)
        if self.optional_settings_path:
            init_params.optional_settings_path = str(self.optional_settings_path)

        err = self.zed.open(init_params)
        if err != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera {self.device_id}: {err}")

        self.runtime_parameters = self._sl.RuntimeParameters()
        self.image = self._sl.Mat()
        self.depth = self._sl.Mat()
        logger.info("ZED camera %s connected successfully", self.device_id)

    def _select_resolution(self, init_params):
        requested = (self.width, self.height)
        for dims, resolution_name in self._RESOLUTION_MAP.items():
            if requested == dims:
                init_params.camera_resolution = getattr(
                    self._sl.RESOLUTION, resolution_name
                )
                return init_params.camera_resolution

        # Default to HD720 if unsupported
        logger.warning(
            "Resolution %sx%s not directly supported. Falling back to HD720.",
            self.width,
            self.height,
        )
        init_params.camera_resolution = self._sl.RESOLUTION.HD720
        return init_params.camera_resolution

    def _get_depth_mode(self):
        depth_mode = self.depth_mode if self.depth_mode in self._DEPTH_MODES else "PERFORMANCE"
        return getattr(self._sl.DEPTH_MODE, depth_mode)

    def capture_frame(self) -> CameraFrame:
        if self.zed is None or self.runtime_parameters is None:
            raise RuntimeError(f"Not connected to ZED camera {self.device_id}")

        if self.zed.grab(self.runtime_parameters) != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab frame from ZED camera {self.device_id}")

        timestamp = time.time()
        self.zed.retrieve_image(self.image, self._sl.VIEW.LEFT)  # bgra
        bgr = self.image.get_data()[:, :, :3]  # Remove alpha channel if present
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_bytes = None
        if self.depth_enabled:
            self.zed.retrieve_measure(self.depth, self._sl.MEASURE.DEPTH)
            depth_arr = self.depth.get_data().copy()
            self.latest_depth = depth_arr
            depth_bytes = depth_arr.astype(np.float32, copy=False).tobytes()

        return {
            "timestamp": timestamp,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "channels": int(rgb.shape[2]),
            "rgb_data": rgb.tobytes(),
            "depth_data": depth_bytes,
        }

    def capture_image(self) -> CameraFrame:
        return self.capture_frame()

    def get_camera_information(self) -> dict:
        if self.zed is None:
            raise RuntimeError("Camera not connected")

        camera_info = self.zed.get_camera_information()
        calib_params = camera_info.camera_configuration.calibration_parameters
        return {
            "serial_number": camera_info.serial_number,
            "camera_model": str(camera_info.camera_model),
            "fx": calib_params.left_cam.fx,
            "fy": calib_params.left_cam.fy,
            "cx": calib_params.left_cam.cx,
            "cy": calib_params.left_cam.cy,
            "baseline": calib_params.get_camera_baseline(),
        }

    def get_latest_depth(self) -> Optional[np.ndarray]:
        return None if self.latest_depth is None else self.latest_depth.copy()

    def show_preview_rgbd(self, frame: CameraFrame) -> None:
        if frame["depth_data"] is None:
            self.show_preview_rgb(frame)
            return

        rgb = np.frombuffer(frame["rgb_data"], dtype=np.uint8).reshape(
            (frame["height"], frame["width"], frame["channels"])
        )
        depth = np.frombuffer(frame["depth_data"], dtype=np.float32).reshape(
            (frame["height"], frame["width"])
        )

        depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero = depth_vis[depth_vis > 0]
        max_val = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
        max_val = max(max_val, 1e-6)
        depth_vis = np.clip(depth_vis, 0.0, max_val)
        depth_vis = (depth_vis / max_val * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        cv2.imshow(f"{self.device_name} RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow(f"{self.device_name} Depth", depth_color)
        cv2.waitKey(1)

    def close(self) -> None:
        if self.zed is not None:
            self.zed.close()
            self.zed = None
        print(f"ZED camera {self.device_id} closed.")

    @staticmethod
    def get_devices(
        amount: int = -1,
        topics: List[str] | None = None,
        config_path: str = "configs/config.yaml",
        topic_prefix: str = "zed",
        height: int = 720,
        width: int = 1280,
        zlc_config: Optional[str] = None,
        **kwargs,
    ) -> List["ZED"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found.
          Leaving out `amount` or `amount = -1` returns all instances.
        - `topics` (List[str] | None): Optional topic names per device. If None, auto-load from config.
        - `config_path` (str): Path to global config containing `camera_streams` with topics (default: configs/config.yaml).
        - `topic_prefix` (str): Fallback prefix when generating topic names, if not found in config (default: "zed").
        - `height` (int): Pixel-height of captured frames. Default: `720`
        - `width` (int): Pixel-width of captured frames. Default: `1280`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[ZED]): List of found devices. If no devices are found, `[]` is returned.
        """
        try:
            sl = ZED._load_sdk()
        except ImportError:
            logger.error(
                "pyzed package not found. Please install the ZED SDK and pyzed package."
            )
            return []

        # Attempt to load topic list from config if not explicitly provided.
        if topics is None:
            topics = ZED._load_topics_from_config(config_path=config_path, prefix=topic_prefix)

        cam_list = sl.Camera.get_device_list()
        devices: List[ZED] = []
        for idx, device in enumerate(cam_list):
            if amount != -1 and idx >= amount:
                break
            topic = topics[idx] if topics and idx < len(topics) else f"{topic_prefix}_{idx}"
            devices.append(
                ZED(
                    device_id=device.serial_number,
                    publish_topic=topic,
                    height=height,
                    width=width,
                    zlc_config=zlc_config,
                    **kwargs,
                )
            )
        return devices

    @staticmethod
    def _load_topics_from_config(config_path: str = "configs/config.yaml", prefix: str = "zed") -> List[str]:
        """Load camera topics from a YAML config's `camera_streams` section."""
        cfg_path = Path(config_path)
        if not cfg_path.is_file():
            logger.warning("Config file %s not found; falling back to generated topics", cfg_path)
            return []

        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to read %s: %s", cfg_path, exc)
            return []

        streams = data.get("camera_streams", {})
        topics: List[str] = []
        for name, cfg in streams.items():
            if prefix and not name.startswith(prefix):
                continue
            topic = cfg.get("topic")
            if topic:
                topics.append(topic)
        if not topics:
            logger.warning("No camera topics found in %s; using generated names", cfg_path)
        return topics


if __name__ == "__main__":
    camera = ZED(
        device_id="30414018",
        publish_topic="zed_camera",
        zlc_config="configs/zed_zlc.yaml",
    )
    try:
        for _ in range(10):
            frame = camera.capture_frame()
            logger.info(
                "Captured frame ts=%s size=%sx%s",
                frame["timestamp"],
                frame["width"],
                frame["height"],
            )
    finally:
        camera.close()
