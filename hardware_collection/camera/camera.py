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
    frame_id: int
    timestamp: float  # Timestamp from CameraHeader (seconds)
    system_time_us: int = field(default_factory=lambda: int(time.time() * 1e6))  # Microseconds since epoch (absolute time)
    system_time_mono_us: int = field(default_factory=lambda: int(time.monotonic() * 1e6))  # Monotonic time in microseconds (resistant to clock adjustments)
    system_time_iso: str = field(default_factory=lambda: datetime.now().isoformat())  # ISO 8601 format for readability
    camera_name: Optional[str] = None  # Optional camera identifier
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_header(cls, header: CameraHeader, system_time_us: Optional[int] = None,
                   system_time_mono_us: Optional[int] = None,
                   system_time_iso: Optional[str] = None,
                   camera_name: Optional[str] = None) -> "FrameMetadata":
        """Create metadata from a CameraHeader.
        
        Args:
            header (CameraHeader): The camera header containing frame info.
            system_time_us (int, optional): System time in microseconds since epoch (absolute). 
                                           If None, uses current time.
            system_time_mono_us (int, optional): Monotonic time in microseconds (clock-adjustment resistant).
                                                If None, uses current monotonic time.
            system_time_iso (str, optional): ISO 8601 formatted system time for readability.
                                            If None, uses current time.
            camera_name (str, optional): Name of the camera.
        
        Returns:
            FrameMetadata: The created metadata.
        """
        if system_time_us is None:
            system_time_us = int(time.time() * 1e6)
        
        if system_time_mono_us is None:
            system_time_mono_us = int(time.monotonic() * 1e6)
        
        if system_time_iso is None:
            system_time_iso = datetime.now().isoformat()
        
        return cls(
            width=header.width,
            height=header.height,
            channels=header.channels,
            frame_id=header.frame_id,
            timestamp=header.timestamp,
            system_time_us=system_time_us,
            system_time_mono_us=system_time_mono_us,
            system_time_iso=system_time_iso,
            camera_name=camera_name
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save metadata to a JSON file.
        
        Args:
            filepath (str): Path to save the metadata JSON file.
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class CameraFrame:
    def __init__(self, header: CameraHeader, image_data: cv2.Mat, 
                 system_time_us: Optional[int] = None, 
                 system_time_mono_us: Optional[int] = None,
                 system_time_iso: Optional[str] = None,
                 camera_name: Optional[str] = None):
        self.header = header
        self.image_data = image_data
        self.image_bytes = image_data.tobytes()
        
        # Capture system time when frame is created
        if system_time_us is None:
            system_time_us = int(time.time() * 1e6)
        if system_time_mono_us is None:
            system_time_mono_us = int(time.monotonic() * 1e6)
        if system_time_iso is None:
            system_time_iso = datetime.now().isoformat()
        
        # Create metadata for this frame
        self.metadata = FrameMetadata.from_header(
            header, 
            system_time_us=system_time_us,
            system_time_mono_us=system_time_mono_us,
            system_time_iso=system_time_iso,
            camera_name=camera_name
        )

    def to_bytes(self) -> List[bytes]:
        return [self.header.to_bytes(), self.image_bytes]

    def to_bytes_combined(self) -> bytes:
        """Combine header and image into single binary payload for pyzlc."""
        header_bytes = self.header.to_bytes()
        return header_bytes + self.image_bytes

    @staticmethod
    def from_bytes(data: bytes):
        """Reconstruct frame from combined binary data."""
        header_size = struct.calcsize(CameraHeader.STRUCT_FORMAT)
        header = CameraHeader.from_bytes(data[:header_size])
        image_bytes = data[header_size:]
        image_data = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            (header.height, header.width, header.channels)
        )
        return CameraFrame(header, image_data)
    
    def save_image(self, filepath: str) -> None:
        """Save the image to a file, converting from RGB to BGR if needed for OpenCV."""
        bgr_image = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr_image)
    
    def save_metadata(self, filepath: str) -> None:
        """Save metadata to a JSON file.
        
        Args:
            filepath (str): Path to save the metadata JSON file.
        """
        self.metadata.save_to_file(filepath)
    
    def get_metadata_dict(self) -> dict:
        """Get metadata as a dictionary.
        
        Returns:
            dict: Metadata dictionary.
        """
        return self.metadata.to_dict()


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
    
    
