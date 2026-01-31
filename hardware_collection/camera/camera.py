from __future__ import annotations
import abc
from typing import TypedDict, Optional
import cv2
import numpy as np

from ..core.abstract_hardware import AbstractHardware

class CameraFrame(TypedDict):
    timestamp: float
    width: int
    height: int
    channels: int
    rgb_data: bytes
    depth_data: Optional[bytes]

class AbstractCamera(AbstractHardware):
    """Camera hardware component."""

    def __init__(
        self,
        device_name: str,
        width: int,
        height: int,
        show_preview: bool,
        depth: bool,
        zlc_config: str
        ) -> None:
        """Initialize the camera hardware component.

        Args:
            device_name (str): Topic name to publish frame data via ZeroLanCom.
            show_preview (bool): Whether to show a preview of the captured images.
        """
        super().__init__(device_name=device_name, config_path=zlc_config)
        self.width = width
        self.height = height
        self.show_preview = show_preview
        self.depth = depth


    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError("Subclasses must implement initialize method.")

    def publish_frame(self) -> None:
        """Publish the captured image frame over ZeroLanCom as binary data.

        Args:
            frame (CameraFrame): The captured image frame.
        """
        frame = self.capture_frame()
        self.publisher.publish(frame)
        if not self.show_preview:
            return
        if self.depth:
            self.show_preview_rgbd(frame)
        else:
            self.show_preview_rgb(frame)


    def show_preview_rgb(self, frame: CameraFrame) -> None:
        """Show a preview of the RGB image frame using OpenCV.

        Args:
            frame (CameraFrame): The captured image frame.
        """
        rgb_array = np.frombuffer(frame["rgb_data"], dtype=np.uint8).reshape(
            (frame["height"], frame["width"], frame["channels"])
        )
        cv2.imshow(f"{self.device_name} Preview", rgb_array)
        cv2.waitKey(1)

    # TODO: implement depth preview
    def show_preview_rgbd(self, frame: CameraFrame) -> None:
        raise NotImplementedError("Subclasses must implement show_preview_rgbd method.")


    @abc.abstractmethod
    def capture_frame(self) -> CameraFrame:
        """Get sensor data from the camera.

        Returns:
            CameraFrame: The sensor data.
        """
        raise NotImplementedError("Subclasses must implement capture_frame method.")
    
    
