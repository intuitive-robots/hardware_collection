import cv2
import numpy as np
from .camera import CameraHeader, CameraFrame, AbstractCamera


class OpenCVCamera(AbstractCamera):
    """
    A concrete implementation of AbstractCamera using OpenCV VideoCapture.
    """

    def __init__(self, address: str, camera_id: int = 0, show_preview: bool = False):
        """
        Args:
            address (str): ZeroMQ address to publish to.
            camera_id (int): OpenCV camera index (default: 0)
            show_preview (bool): Whether to show debug preview window.
        """
        super().__init__(address=address, show_preview=show_preview)
        self.camera_id = camera_id
        self.cap = None
        self.frame_id_counter = 0

    def initialize(self) -> None:
        """Initialize OpenCV camera."""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera ID {self.camera_id}")

        # Read default resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[OpenCVCamera] Camera initialized: {self.width}x{self.height}")

    def capture_image(self) -> CameraFrame:
        """Capture a frame from the OpenCV camera."""
        success, frame = self.cap.read()
        if not success:
            raise RuntimeError("Failed to capture image from OpenCV camera.")

        # Ensure BGR format (OpenCV default)
        height, width, channels = frame.shape

        header = CameraHeader(
            width=width,
            height=height,
            channels=channels,
            frame_id=self.frame_id_counter,
        )

        self.frame_id_counter += 1

        return CameraFrame(header, frame)

    def close(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
