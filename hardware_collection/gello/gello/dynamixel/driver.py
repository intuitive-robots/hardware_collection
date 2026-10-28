import os
import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence

import numpy as np
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_read import GroupSyncRead
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_write import GroupSyncWrite
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import PacketHandler
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import PortHandler
from hardware_collection.gello.third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
# tried after the requested baudrate when the servos do not answer (GELLO default is 57600)
FALLBACK_BAUDRATES = (1000000, 57600, 2000000, 3000000, 4000000, 115200)
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = False

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self, ids: Sequence[int], port: str = "/dev/ttyUSB0", baudrate: int = 57600
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._ids = ids
        self._joint_angles = None
        # (raw angles, monotonic read time, read count) of the latest bus read, swapped as one object
        self._sample = None
        self._sample_count = 0
        self._lock = Lock()

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        self.baudrate = self._find_baudrate(baudrate)
        self._set_low_latency(port)

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

        self._stop_thread = Event()
        self._start_reading_thread()

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")

        for dxl_id, angle in zip(self._ids, joint_angles):
            # Convert the angle to the appropriate value for the servo
            position_value = int(angle * 2048 / np.pi)

            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._groupSyncWrite.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")

        # Clear syncwrite parameter storage
        self._groupSyncWrite.clearParam()

    def _find_baudrate(self, preferred: int) -> int:
        """Open the bus at the baudrate the servos answer on, trying `preferred` first.

        The servo baudrate is stored in their EEPROM (see scripts/gello_set_baudrate.py), so
        callers do not need to know whether the GELLO was switched from 57600 to 1 Mbps.
        """
        for baud in [preferred] + [b for b in FALLBACK_BAUDRATES if b != preferred]:
            if not self._portHandler.setBaudRate(baud):
                raise RuntimeError(f"Failed to change the baudrate, {baud}")
            _, res, err = self._packetHandler.ping(self._portHandler, self._ids[0])
            if res == COMM_SUCCESS and err == 0:
                if baud != preferred:
                    print(f"Dynamixel ID {self._ids[0]} answers at {baud} baud (not {preferred}), using {baud}")
                return baud
        raise RuntimeError(f"Dynamixel ID {self._ids[0]} does not answer at any of {[preferred, *FALLBACK_BAUDRATES]}")

    def _set_low_latency(self, port: str) -> None:
        """Drop the FTDI latency timer from 16 ms to 1 ms.

        With the default 16 ms every bus read waits for the USB adapter to flush, which limits
        GELLO to ~20 samples/s. The low_latency serial flag makes ftdi_sio use 1 ms; unlike
        writing /sys/.../latency_timer or a udev rule this works as normal user inside the
        container. It holds until the adapter is re-plugged, so it is set on every open.
        """
        try:
            self._portHandler.ser.set_low_latency_mode(True)
        except Exception as e:  # not a Linux tty or driver without support
            print(f"warning: could not set low latency mode on {port}: {e}")
            return
        tty = os.path.basename(os.path.realpath(port))
        try:
            with open(f"/sys/bus/usb-serial/devices/{tty}/latency_timer") as f:
                print(f"{port}: FTDI latency timer {f.read().strip()} ms")
        except OSError:
            pass

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )

        self._torque_enabled = enable

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_angles(self):
        # Continuously read joint angles and update the joint_angles array
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            with self._lock:
                _joint_angles = np.zeros(len(self._ids), dtype=int)
                dxl_comm_result = self._groupSyncRead.txRxPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"warning, comm failed: {dxl_comm_result}")
                    continue
                for i, dxl_id in enumerate(self._ids):
                    if self._groupSyncRead.isAvailable(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    ):
                        angle = self._groupSyncRead.getData(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        )
                        angle = np.int32(np.uint32(angle))
                        _joint_angles[i] = angle
                    else:
                        raise RuntimeError(
                            f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
                        )
                self._joint_angles = _joint_angles
                self._sample_count += 1
                self._sample = (_joint_angles, time.monotonic(), self._sample_count)
            # self._groupSyncRead.clearParam() # TODO what does this do? should i add it

    def get_joints(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        while self._joint_angles is None:
            time.sleep(0.1)
        # with self._lock:
        _j = self._joint_angles.copy()
        return _j / 2048.0 * np.pi

    def get_joints_stamped(self) -> tuple[np.ndarray, float, int]:
        """Latest joint angles [rad] with the monotonic time and running count of their bus read."""
        while self._sample is None:
            time.sleep(0.01)
        angles, stamp, count = self._sample
        return angles / 2048.0 * np.pi, stamp, count

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self._portHandler.closePort()


def main():
    # Set the port, baudrate, and servo IDs
    ids = [1]

    # Create a DynamixelDriver instance
    try:
        driver = DynamixelDriver(ids)
    except FileNotFoundError:
        driver = DynamixelDriver(ids, port="/dev/cu.usbserial-FT7WBMUB")

    # Test setting torque mode
    driver.set_torque_mode(True)
    driver.set_torque_mode(False)

    # Test reading the joint angles
    try:
        while True:
            joint_angles = driver.get_joints()
            print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
    except KeyboardInterrupt:
        driver.close()


if __name__ == "__main__":
    main()  # Test the driver
