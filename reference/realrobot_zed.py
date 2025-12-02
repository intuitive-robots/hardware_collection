import time
import numpy as np
import logging

from real_robot_env.robot.hardware_cameras import DiscreteCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZED(DiscreteCamera):
    """
    Wrapper that implements boilerplate code for ZED cameras from Stereolabs.

    This class uses the ZED SDK to access RGB and depth data with CUDA acceleration.

    This class inherits its functions from `real_robot_env.robot.hardware_cameras.DiscreteCamera`.

    Note: Requires the ZED SDK and pyzed package to be installed.
    """

    def __init__(
        self,
        device_id,
        name=None,
        height=720,
        width=1280,
        fps=30,
        depth_mode="PERFORMANCE",
        start_frame_latency=0,
    ):
        """
        Initialize ZED camera with SDK support.

        Parameters:
        -----------
        device_id : str or int
            Serial number of the ZED camera (e.g., "30414018")
        name : str, optional
            Name for the camera instance
        height : int
            Pixel-height of captured frames. Default: 720
        width : int
            Pixel-width of captured frames. Default: 1280
        fps : int
            Frames per second. Default: 30
        depth_mode : str
            Depth mode: "NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL". Default: "PERFORMANCE"
        start_frame_latency : int
            Latency for frame capture start
        """
        super().__init__(
            device_id,
            name if name else f"ZED_{device_id}",
            height,
            width,
            start_frame_latency,
        )
        self.fps = fps
        self.depth_mode = depth_mode
        self.zed = None
        self.runtime_parameters = None

    def _setup_connect(self):
        super()._setup_connect()

        try:
            import pyzed.sl as sl

            self.sl = sl
        except (ImportError, AttributeError):
            try:
                import pyzed as sl

                self.sl = sl
            except ImportError:
                raise ImportError(
                    "pyzed package not found. Please install the ZED SDK and pyzed package. "
                    "Visit: https://www.stereolabs.com/developers/release/"
                )

        # Create a Camera object
        self.zed = self.sl.Camera()

        # Create InitParameters object and set configuration parameters
        init_params = self.sl.InitParameters()

        # Set the serial number to connect to a specific camera
        init_params.camera_serial_number = int(self.device_id)

        # Set resolution
        if self.width == 2208 and self.height == 1242:
            init_params.camera_resolution = self.sl.RESOLUTION.HD2K
        elif self.width == 1920 and self.height == 1080:
            init_params.camera_resolution = self.sl.RESOLUTION.HD1080
        elif self.width == 1280 and self.height == 720:
            init_params.camera_resolution = self.sl.RESOLUTION.HD720
        elif self.width == 672 and self.height == 376:
            init_params.camera_resolution = self.sl.RESOLUTION.VGA
        else:
            # Default to HD720 and resize later
            init_params.camera_resolution = self.sl.RESOLUTION.HD720
            logger.warning(
                f"Resolution {self.width}x{self.height} not directly supported. "
                "Using HD720 and will resize."
            )

        # Set FPS
        init_params.camera_fps = self.fps

        # Set depth mode
        depth_modes = {
            "NONE": self.sl.DEPTH_MODE.NONE,
            "PERFORMANCE": self.sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": self.sl.DEPTH_MODE.QUALITY,
            "ULTRA": self.sl.DEPTH_MODE.ULTRA,
            "NEURAL": self.sl.DEPTH_MODE.NEURAL,
        }
        init_params.depth_mode = depth_modes.get(
            self.depth_mode, self.sl.DEPTH_MODE.PERFORMANCE
        )

        # Set units to millimeters
        init_params.coordinate_units = self.sl.UNIT.MILLIMETER

        # Open the camera
        err = self.zed.open(init_params)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera {self.device_id}: {err}")

        # Create runtime parameters
        self.runtime_parameters = self.sl.RuntimeParameters()

        # Create Mat objects to store images
        self.image = self.sl.Mat()
        self.depth = self.sl.Mat()

        logger.info(f"ZED camera {self.device_id} connected successfully")

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd': depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format
          `{'time': float, 'rgb': NDArray[uint8], 'd': NDArray[float32]}`.
        """
        if self.zed is None:
            raise RuntimeError(f"Not connected to {self.name}")

        # Grab a new frame
        if self.zed.grab(self.runtime_parameters) == self.sl.ERROR_CODE.SUCCESS:
            timestamp = time.time()

            # Retrieve left image (RGB)
            self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
            rgb = self.image.get_data()[:, :, :3]  # Remove alpha channel if present

            # Retrieve depth map
            self.zed.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)
            depth = self.depth.get_data()

            return {"time": timestamp, "rgb": rgb, "d": depth}
        else:
            raise RuntimeError(f"Failed to grab frame from ZED camera {self.device_id}")

    def get_camera_information(self):
        """
        Get camera calibration and information.

        Returns:
        --------
        dict: Dictionary containing camera parameters
        """
        if self.zed is None:
            raise RuntimeError("Camera not connected")

        camera_info = self.zed.get_camera_information()
        calib_params = camera_info.camera_configuration.calibration_parameters

        return {
            "serial_number": camera_info.serial_number,
            "camera_model": str(camera_info.camera_model),
            "fx": calib_params.left_cam.fx,
            "fy": calib_params.left_cam.fy,
            "cx": calib_params.left_cam.cx,
            "cy": calib_params.left_cam.cy,
            "baseline": calib_params.get_camera_baseline(),
        }

    def close(self):
        """Close the camera connection."""
        success = super().close()
        if self.zed is not None:
            self.zed.close()
            self.zed = None
        return success

    @staticmethod
    def get_devices(
        amount=-1, height: int = 720, width: int = 1280, **kwargs
    ) -> list["ZED"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found.
          Leaving out `amount` or `amount = -1` returns all instances.
        - `height` (int): Pixel-height of captured frames. Default: `720`
        - `width` (int): Pixel-width of captured frames. Default: `1280`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[ZED]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(ZED, ZED).get_devices(
            amount, height=height, width=width, device_type="ZED", **kwargs
        )
        try:
            import pyzed.sl as sl
        except (ImportError, AttributeError):
            try:
                import pyzed as sl
            except ImportError:
                logger.error(
                    "pyzed package not found. Please install the ZED SDK and pyzed package."
                )
                return []

        # Get list of connected ZED cameras
        cam_list = sl.Camera.get_device_list()
        cams = []
        counter = 0

        for device in cam_list:
            if amount != -1 and counter >= amount:
                break
            cam = ZED(device.serial_number, height=height, width=width, **kwargs)
            cams.append(cam)
            counter += 1

        return cams


if __name__ == "__main__":
    # Test the ZED camera
    cam = ZED(device_id="30414018")
    cam.connect()
    print("Camera info:", cam.get_camera_information())

    for i in range(10):
        sensors = cam.get_sensors()
        print(
            f"Frame {i}: RGB shape={sensors['rgb'].shape}, Depth shape={sensors['d'].shape}"
        )

    cam.close()