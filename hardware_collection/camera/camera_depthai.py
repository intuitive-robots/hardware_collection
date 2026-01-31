from __future__ import annotations
import os
import time
import depthai as dai  # pylint: disable=no-member
import enum
import yaml
import argparse

from .camera import AbstractCamera, CameraFrame

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
        zlc_config: str = "configs/zlc.yaml"
    ):
        self.camera_type = camera_type
        self.device_id = device_id
        self.name = name or f"DepthAI_{device_id}"
        super().__init__(self.name, width, height, show_preview=True, depth=False, zlc_config=zlc_config)
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
                dai.CameraBoardSocket.CAM_C,
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
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        cam_rgb.setFps(30)

        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        self.device = dai.Device(self.pipeline, dai.UsbSpeed.SUPER)
        assert self.device.getUsbSpeed() == dai.UsbSpeed.SUPER, "DepthAI camera requires USB 3.0 connection."
        
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


    def capture_frame(self) -> CameraFrame:
        """Get sensor data from the DepthAI camera.

        Returns:
            CameraFrame: The sensor data.
        """
        raw_data = self.q_rgb.get().getRaw()
        rgb_mat = raw_data.data.reshape((3, self.height, self.width))
        rgb_mat = rgb_mat.transpose(1, 2, 0)  # HWC format
        frame = CameraFrame(
            height=self.height,
            width=self.width,
            channels=3,
            timestamp=time.time_ns(),
            rgb_data= rgb_mat.tobytes(),
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
    try:
        depthai_camera = DepthAICamera(
            config["device_name"],
            device_id=config["device_id"],
            width=config["width"],
            height=config["height"]
        )
        while True:
            depthai_camera.publish_frame()
    except KeyboardInterrupt:
        print("Exiting...")