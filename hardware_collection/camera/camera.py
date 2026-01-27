from __future__ import annotations
import abc
import time
import struct
import json
from typing import ClassVar, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
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


@dataclass
class FrameMetadata:
    """
    Metadata for a camera frame that can be saved alongside the frame for later correspondence.
    
    This includes:
    - Frame information: width, height, channels, frame_id
    - Timestamps: both float timestamp (from header) and system time (ISO format)
    - Additional fields for correspondence tracking
    """
    
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
        """Save the image to a file.
        
        Args:
            filepath (str): Path to save the image file.
        """
        cv2.imwrite(filepath, self.image_data)
    
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

    def __init__(self, publish_topic: str, show_preview: bool = False, 
                 camera_name: Optional[str] = None) -> None:
        """Initialize the camera hardware component.

        Args:
            publish_topic (str): Topic name to publish frame data via ZeroLanCom.
            show_preview (bool): Whether to show a preview of the captured images.
            camera_name (str, optional): Name of the camera for metadata tracking.
        """
        super().__init__(publish_topic=publish_topic)
        self.width = None
        self.height = None
        self.show_preview = show_preview
        self.camera_name = camera_name

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
    
    
