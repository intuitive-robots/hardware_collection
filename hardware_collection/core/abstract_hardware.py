import abc
from typing import Optional
import pyzlc


class AbstractHardware:
    """Abstract base class for hardware components using ZeroLanCom Publisher."""

    def __init__(self, publish_topic: str):
        """Initialize with pyzlc Publisher.

        Args:
            publish_topic (str): Topic name to publish data via ZeroLanCom.
        """
        self.publish_topic = publish_topic
        self.publisher: Optional[pyzlc.Publisher] = None
        
        try:
            self.publisher = pyzlc.Publisher(publish_topic)
        except Exception as e:
            print(f"Warning: Failed to create publisher for topic '{publish_topic}': {e}")

    def close_sockets(self) -> None:
        """Close any allocated ZeroLanCom publishers."""
        self.publisher = None

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the hardware component."""
        raise NotImplementedError("Subclasses must implement initialize method.")

