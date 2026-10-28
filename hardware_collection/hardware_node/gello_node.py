import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pyzlc
import tyro

from hardware_collection.gello.gello.agents.gello_agent import GelloAgent

from hardware_collection.core.abstract_hardware import AbstractHardware

#todo: reconstuct config
@dataclass
class Args:
    node_name: str = "gello"
    ip: str = "192.168.1.1"
    hardware_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EVRT-if00-port0" #check the actual config
    baudrate: int = 1000000
    """Preferred Dynamixel bus baudrate; the driver falls back to the rate the servos answer on
    (57600 until scripts/gello_set_baudrate.py was run)."""
    velocity_cutoff_hz: float = 10.0
    """Low-pass cutoff of the finite-difference joint velocity sent as feed-forward."""
    poll_period_s: float = 0.0005
    """How often to check the driver for a new bus read."""


class GelloNode:
    """Publishes one message per new Dynamixel bus read (not per loop iteration)."""

    def __init__(self, name: str, agent: GelloAgent, velocity_cutoff_hz: float = 10.0):
        self._name = name
        self._agent = agent
        self._velocity_cutoff_hz = velocity_cutoff_hz
        self._stop_event = threading.Event()
        self._state_thread: Optional[threading.Thread] = None
        self._last_count = 0
        self._last_pos: Optional[np.ndarray] = None
        self._last_stamp = 0.0
        self._velocity: Optional[np.ndarray] = None

        self._arm_state_pub = pyzlc.Publisher(f"{name}/gello_arm_state")
        pyzlc.info(f"[GelloNode] Initialized publishers for '{name}'/gello_arm_state")
        self._gripper_state_pub = pyzlc.Publisher(f"{name}/gello_gripper_state")
        pyzlc.info(f"[GelloNode] Initialized publishers for '{name}'/gello_gripper_state")

    def poll(self) -> bool:
        """Publish the latest bus read if it is new. Returns True if something was published."""
        state, stamp, count = self._agent._robot.get_joint_state_stamped()
        if count == self._last_count:
            return False
        self._last_count = count
        joints = np.asarray(state[:-1], dtype=float)

        # finite difference between bus reads, low-passed against encoder quantization noise
        if self._last_pos is None or stamp <= self._last_stamp:
            self._velocity = np.zeros_like(joints)
        else:
            dt = stamp - self._last_stamp
            raw = (joints - self._last_pos) / dt
            a = 1.0 - np.exp(-2 * np.pi * self._velocity_cutoff_hz * dt)
            self._velocity = self._velocity + a * (raw - self._velocity)
        self._last_pos, self._last_stamp = joints, stamp

        self._arm_state_pub.publish(self._build_arm_state(joints, stamp, count))
        self._gripper_state_pub.publish({"gripper": float(state[-1])})
        return True

    def _build_arm_state(self, joints: np.ndarray, stamp: float, count: int) -> Dict[str, List[float]]:
        return {
            "joint_state": joints.tolist(),
            "joint_velocity": self._velocity.tolist(),
            # monotonic time of the bus read on the GELLO host and a running sample number
            "stamp": stamp,
            "seq": count,
        }

def main():
    args = tyro.cli(Args)
    pyzlc.init(args.node_name, args.ip, group_name="DroidGroup",group_port=7730)
    agent = GelloAgent(port=args.hardware_port, baudrate=args.baudrate)
    gello_node = GelloNode(args.node_name, agent, velocity_cutoff_hz=args.velocity_cutoff_hz)
    pyzlc.info(f"[GelloNode] ZeroLanCom node '{args.node_name}' started on {args.ip}")
    try:
        while True:
            if not gello_node.poll():
                pyzlc.sleep(args.poll_period_s)
    except Exception as exc:  # pragma: no cover - runtime feedback only
        pyzlc.error("Publisher stopped due to error: %s", exc)
        return 1
    finally:
        pyzlc.info("Gello publisher exiting")
        return 0


if __name__ == "__main__":
    main()
