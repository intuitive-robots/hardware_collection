from __future__ import annotations
import abc
import time
import struct
from typing import ClassVar, List, Optional
from dataclasses import dataclass, field
import cv2
import numpy as np

from hardware_collection.core.abstract_hardware import AbstractHardware


@dataclass
class CameraHeader:
    """
    Represents the metadata of a camera frame.
    Uses `struct` to pack data into a compact, fixed-size binary format.

    Binary Layout (little-endian "<"):
        - width (I):     4 bytes, unsigned int
        - height (I):    4 bytes, unsigned int
        - channels (B):  1 byte, unsigned char
        - timestamp (d): 8 bytes, double
        - frame_id (Q):  8 bytes, unsigned long long

    Total: 29 bytes (you can pad to 32 for alignment if needed).
    """

    STRUCT_FORMAT: ClassVar[str] = "<IIBdQ"  # Little-endian layout
    STRUCT_SIZE: ClassVar[int] = struct.calcsize(STRUCT_FORMAT)

    width: int
    height: int
    channels: int
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    def to_bytes(self) -> bytes:
        """
        Serialize the header into a fixed-length byte sequence.

        Returns:
            bytes: Packed binary representation of the header.
        """
        return struct.pack(
            self.STRUCT_FORMAT,
            self.width,
            self.height,
            self.channels,
            self.timestamp,
            self.frame_id
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CameraHeader":
        """
        Deserialize bytes into a CameraHeader object.

        Args:
            data (bytes): The packed header bytes.

        Returns:
            CameraHeader: The reconstructed header.
        """
        width, height, channels, timestamp, frame_id = struct.unpack(
            cls.STRUCT_FORMAT,
            data[:cls.STRUCT_SIZE]
        )
        return cls(width, height, channels, timestamp, frame_id)

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f"<CameraHeader "
            f"{self.width}x{self.height}x{self.channels} "
            f"id={self.frame_id} t={self.timestamp:.3f}>"
        )


class CameraFrame:
    def __init__(self, header: CameraHeader, image_data: cv2.Mat):
        self.header = header
        self.image_data = image_data
        self.image_bytes = image_data.tobytes()

    def to_bytes(self) -> List[bytes]:
        """Legacy method for ZMQ compatibility (multipart)."""
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


class AbstractCamera(AbstractHardware):
    """Camera hardware component."""

    def __init__(self, publish_topic: str, show_preview: bool = False) -> None:
        """Initialize the camera hardware component.

        Args:
            publish_topic (str): Topic name to publish frame data via ZeroLanCom.
            show_preview (bool): Whether to show a preview of the captured images.
        """
        super().__init__(publish_topic=publish_topic)
        self.width = None
        self.height = None
        self.show_preview = show_preview

    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError("Subclasses must implement initialize method.")

    def publish_image(self, frame: Optional[CameraFrame] = None) -> None:
        """Publish the captured image frame over ZeroLanCom as binary data.

        Args:
            frame (CameraFrame): The captured image frame.
        """
        if frame is None:
            frame = self.capture_image()
        
        if self.publisher is not None:
            try:
                self.publisher.publish(frame.to_bytes_combined())
            except Exception as e:
                print(f"Warning: Failed to publish frame on topic '{self.publish_topic}': {e}")

    @abc.abstractmethod
    def capture_image(self) -> CameraFrame:
        """Get sensor data from the camera.

        Returns:
            CameraFrame: The sensor data.
        """
        raise NotImplementedError("Subclasses must implement capture_image method.")
    
    
