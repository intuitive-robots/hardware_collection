from __future__ import annotations
import argparse
import enum
import logging
import os
import time

import cv2
import depthai as dai  # pylint: disable=no-member
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
        show_preview: bool = True,
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
            show_preview=show_preview,
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


    def capture_frame(self) -> CameraFrame:
        """Get sensor data from the DepthAI camera.

        Returns:
            CameraFrame: The sensor data.
        """
        raw_data = self.q_rgb.get().getRaw()
        rgb_mat = raw_data.data.reshape((3, self.height, self.width))
        bgr_mat = rgb_mat[[2, 1, 0], :, :]  # Convert RGB to BGR for OpenCV
        bgr_mat = bgr_mat.transpose(1, 2, 0)  # HWC RGB format
        frame = CameraFrame(
            height=self.height,
            width=self.width,
            channels=3,
            timestamp=time.time_ns(),
            rgb_data= bgr_mat.tobytes(),
            depth_data=None,
        )
        return frame


    @staticmethod
    def get_devices() -> None:
        cam_list = dai.Device.getAllAvailableDevices()
        for device in cam_list:
            print(f"Found DepthAI camera: {device.getMxId()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/zlc_config.yaml", help="Path to ZLC config file.")
    args = parser.parse_args()
    DepthAICamera.get_devices()
    with open(os.path.join(os.getcwd(), args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    depthai_camera = None
    try:
        depthai_camera = DepthAICamera(
            config["device_name"],
            device_id=config["device_id"],
            width=config["width"],
            height=config["height"],
            camera_type=DAICameraType[config.get("camera_type", "OAK_D_LITE")],
            fps=config.get("fps", 30),
            require_usb3=config.get("require_usb3", False),
            show_preview=config.get("show_preview", True),
            zlc_config=config.get("zlc_config_path", "configs/zlc.yaml"),
        )
        while True:
            depthai_camera.publish_frame()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if depthai_camera is not None and depthai_camera.show_preview:
            cv2.destroyAllWindows()
