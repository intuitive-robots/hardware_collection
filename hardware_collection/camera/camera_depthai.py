from __future__ import annotations

import argparse
import enum
import logging
import os
import time

import cv2
import depthai as dai  # pylint: disable=no-member
import numpy as np
import yaml

from .camera import AbstractCamera, CameraFrame

logger = logging.getLogger(__name__)


class DAICameraType(enum.Enum):
    OAK_D = 0
    OAK_D_LITE = 1
    OAK_D_SR = 2


class DepthAICamera(AbstractCamera):
    """DepthAI camera hardware component."""

    def __init__(
        self,
        name: str,
        device_id: str,
        height: int = 640,
        width: int = 480,
        camera_type: DAICameraType = DAICameraType.OAK_D_LITE,
        fps: int = 30,
        require_usb3: bool = False,
        zlc_config: str = "configs/zlc.yaml",
    ):
        self.camera_type = camera_type
        self.device_id = device_id
        self.name = name or f"DepthAI_{device_id}"
        self.fps = fps
        self.require_usb3 = require_usb3
        super().__init__(
            self.name,
            width,
            height,
            show_preview=True,
            depth=False,
            zlc_config=zlc_config,
        )
        self.initialize()

    def initialize(self) -> None:
        """Initialize the DepthAI camera hardware."""
        if self.camera_type in [DAICameraType.OAK_D, DAICameraType.OAK_D_LITE]:
            _, board_socket, resolution = (
                dai.node.ColorCamera,
                dai.CameraBoardSocket.CAM_A,
                dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            )
        elif self.camera_type == DAICameraType.OAK_D_SR:
            _, board_socket, resolution = (
                dai.node.ColorCamera,
                dai.CameraBoardSocket.CAM_B,
                dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            )
        else:
            raise ValueError("Unsupported DepthAI camera type.")

        self.device_info = dai.DeviceInfo(self.device_id)

        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(board_socket)
        cam_rgb.setResolution(resolution)

        cam_rgb.setPreviewSize(self.width, self.height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        cam_rgb.setFps(self.fps)

        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_resolution = self._get_mono_resolution()
        mono_left.setResolution(mono_resolution)
        mono_right.setResolution(mono_resolution)
        mono_left.setFps(self.fps)
        mono_right.setFps(self.fps)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setConfidenceThreshold(200)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(board_socket)
        stereo.setOutputSize(self.width, self.height)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(xout_depth.input)

        self.device = dai.Device(self.pipeline, self.device_info, dai.UsbSpeed.SUPER)
        self.usb_speed = self.device.getUsbSpeed()
        if self.usb_speed != dai.UsbSpeed.SUPER:
            message = (
                f"DepthAI camera {self.device_id} connected at USB speed "
                f"{self.usb_speed}, not SUPER. Use a USB 3 cable/port for full bandwidth."
            )
            if self.require_usb3:
                raise RuntimeError(message)
            logger.warning("%s Continuing because require_usb3 is false.", message)

        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    def _get_mono_resolution(self):
        """Return a mono resolution supported by this camera model."""
        if self.camera_type == DAICameraType.OAK_D_LITE:
            return dai.MonoCameraProperties.SensorResolution.THE_480_P
        return dai.MonoCameraProperties.SensorResolution.THE_800_P

    def capture_frame(self) -> CameraFrame:
        """Get sensor data from the DepthAI camera.

        Returns:
            CameraFrame: The sensor data.
        """
        rgb_raw = self.q_rgb.get().getRaw()
        rgb_mat = rgb_raw.data.reshape((3, self.height, self.width))
        rgb_mat = rgb_mat.transpose(1, 2, 0)

        depth_mat = self.q_depth.get().getFrame()
        if depth_mat.shape != (self.height, self.width):
            depth_mat = cv2.resize(
                depth_mat,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )

        frame = CameraFrame(
            height=self.height,
            width=self.width,
            channels=3,
            timestamp=time.time(),
            rgb_data=rgb_mat.tobytes(),
            depth_data=depth_mat.astype(np.uint16, copy=False).tobytes(),
        )
        return frame

    def show_preview_rgbd(self, frame: CameraFrame) -> None:
        if frame["depth_data"] is None:
            self.show_preview_rgb(frame)
            return

        rgb = np.frombuffer(frame["rgb_data"], dtype=np.uint8).reshape(
            (frame["height"], frame["width"], frame["channels"])
        )
        depth = np.frombuffer(frame["depth_data"], dtype=np.uint16).reshape(
            (frame["height"], frame["width"])
        )

        depth_vis = np.nan_to_num(
            depth.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        nonzero = depth_vis[depth_vis > 0]
        max_val = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
        max_val = max(max_val, 1e-6)
        depth_vis = np.clip(depth_vis, 0.0, max_val)
        depth_vis = (depth_vis / max_val * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        cv2.imshow(f"{self.device_name} RGB", rgb)
        cv2.imshow(f"{self.device_name} Depth", depth_color)
        cv2.waitKey(1)

    @staticmethod
    def get_devices() -> None:
        cam_list = dai.Device.getAllAvailableDevices()
        for device in cam_list:
            print(f"Found DepthAI camera: {device.getMxId()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/depthai_wrist.yaml",
        help="Path to DepthAI camera config file.",
    )
    args = parser.parse_args()
    DepthAICamera.get_devices()
    with open(os.path.join(os.getcwd(), args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    try:
        depthai_camera = DepthAICamera(
            config["device_name"],
            device_id=config["device_id"],
            width=config["width"],
            height=config["height"],
            camera_type=DAICameraType[config.get("camera_type", "OAK_D_LITE")],
            fps=config.get("fps", 30),
            require_usb3=config.get("require_usb3", False),
            zlc_config=config.get("zlc_config_path", "configs/zlc.yaml"),
        )
        while True:
            depthai_camera.publish_frame()
    except KeyboardInterrupt:
        print("Exiting...")
