from __future__ import annotations
import time
import depthai as dai  # pylint: disable=no-member
import enum
from typing import List
import cv2

from .camera import AbstractCamera, CameraFrame, CameraHeader

class DAICameraType(enum.Enum):
    OAK_D = 0
    OAK_D_LITE = 1
    OAK_D_SR = 2


class DepthAICamera(AbstractCamera):
    """DepthAI camera hardware component."""

    def __init__(
        self,
        device_id,
        publish_topic: str,
        name=None,
        height=512,
        width=512,
        camera_type: DAICameraType = DAICameraType.OAK_D_LITE,
    ):
        self.camera_type = camera_type
        self.device_id = device_id
        self.name = name or f"DepthAI_{device_id}"
        super().__init__(publish_topic=publish_topic)
        self.initialize()

    def initialize(self) -> None:
        """Initialize the DepthAI camera hardware."""
        if self.camera_type in [DAICameraType.OAK_D, DAICameraType.OAK_D_LITE]:
            depthai_cam, board_socket, resolution = (
                dai.node.ColorCamera,
                dai.CameraBoardSocket.CAM_A,
                dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            )
        elif self.camera_type == DAICameraType.OAK_D_SR:
            depthai_cam, board_socket, resolution = (
                dai.node.ColorCamera,
                dai.CameraBoardSocket.CAM_C,
                dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            )
        else:
            raise ValueError("Unsupported DepthAI camera type.")

        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.create(depthai_cam)
        cam_rgb.setBoardSocket(board_socket)
        cam_rgb.setResolution(resolution)
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)  # type: ignore
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        self.device_info = dai.DeviceInfo(self.device_id)

        self.device = dai.Device(self.pipeline.getDeviceConfig(), self.device_info)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)  # type: ignore


    def capture_image(self) -> CameraFrame:
        """Get sensor data from the DepthAI camera.

        Returns:
            CameraFrame: The sensor data.
        """
        bgr_img = self.q_rgb.get().getCvFrame()
        frame = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        frame = CameraFrame(
            header=CameraHeader(
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2],
                timestamp=time.time(),
                frame_id=0,
            ),
            image_data=frame,
        )
        return frame


    @staticmethod
    def get_devices(
        amount=-1,
        topics: List[str] | None = None,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ) -> List[DepthAICamera]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `topics` (List[str] | None): Optional publish topics to use for each device. If omitted, defaults to auto-generated names.
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[DepthAICamera]): List of found devices. If no devices are found, `[]` is returned.
        """

        cam_list = dai.Device.getAllAvailableDevices()
        cams = []
        counter = 0
        for device in cam_list:
            if amount != -1 and counter >= amount:
                break
            topic_name = (
                topics[counter] if topics and counter < len(topics) else f"depthai_{counter}"
            )
            cam = DepthAICamera(
                device.getMxId(),
                publish_topic=topic_name,
                height=height,
                width=width,
            )
            cams.append(cam)
            counter += 1
        return cams
