import abc
import pyzlc
import yaml
import os

class AbstractHardware:
    """Abstract base class for hardware components using ZeroLanCom Publisher."""

    def __init__(self, device_name: str, config_path: str):
        """Initialize with pyzlc Publisher.

        Args:
            device_name (str): Topic name to publish data via ZeroLanCom.
            config_path (str): Path to the ZeroLanCom configuration file.
        """
        self.device_name = device_name
        print(f"Loading ZLC config from: {os.path.join(os.getcwd(), config_path)}")
        with open(os.path.join(os.getcwd(), config_path), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
        pyzlc.init(self.device_name, self.config["local_ip"], group_name=self.config["group_name"])
        self.publisher = pyzlc.Publisher(self.device_name)

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize any required ZeroLanCom nodes."""
        raise NotImplementedError("Subclasses must implement initialize method.")
