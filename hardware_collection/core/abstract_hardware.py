import zmq
import abc


class AbstractHardware:
    """Abstract base class for hardware components using ZeroMQ PUB sockets for communication."""

    def __init__(self, publish_address: str):
        """Initialize the hardware component with a PUB ZeroMQ socket.

        Args:
            publish_address (str): Address to bind the PUB socket for downstream subscribers.
        """
        self._ctx = zmq.Context.instance()

        self.pub_socket = self._ctx.socket(zmq.PUB)
        self.pub_socket.bind(publish_address)

    def close_sockets(self) -> None:
        """Close any allocated ZeroMQ sockets."""
        if getattr(self, "pub_socket", None) is not None:
            self.pub_socket.close(0)
            self.pub_socket = None

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the hardware component."""
        raise NotImplementedError("Subclasses must implement initialize method.")

