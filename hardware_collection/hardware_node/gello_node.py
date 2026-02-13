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
    ip: str = "192.168.0.109"
    hardware_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EVRT-if00-port0" #check the actual config
    state_pub_rate_hz: int = 1000


class GelloNode:
   
    def __init__(self, name: str, agent: GelloAgent, state_pub_rate_hz: int = 50,):
        self._name = name
        self._agent = agent
        self._state_dt_s = int(1 / max(1, state_pub_rate_hz))
        self._stop_event = threading.Event()
        self._state_thread: Optional[threading.Thread] = None

        self._arm_state_pub = pyzlc.Publisher(f"{name}/gello_arm_state")
        pyzlc.info(f"[GelloNode] Initialized publishers for '{name}'/gello_arm_state")
        self._gripper_state_pub = pyzlc.Publisher(f"{name}/gello_gripper_state")
        pyzlc.info(f"[GelloNode] Initialized publishers for '{name}'/gello_gripper_state")
    def _build_arm_state(self) -> Dict[str, List[float]]:
        joints_arr = np.asarray(self._agent._robot.get_joint_state()[:-1], dtype=float).reshape(-1)
        return {"joint_state": joints_arr.tolist()}

    def _build_gripper_state(self) -> Dict[str, float]:
        gripper = float(np.asarray(self._agent._robot.get_joint_state()[-1], dtype=float))
        return {"gripper": gripper}

def main():
    args = tyro.cli(Args)
    pyzlc.init(args.node_name, args.ip, group_name="DroidGroup",group_port=7730)
    agent = GelloAgent(port=args.hardware_port)
    gello_node = GelloNode(args.node_name, agent,
                        state_pub_rate_hz=args.state_pub_rate_hz)
    pyzlc.info(f"[GelloNode] ZeroLanCom node '{args.node_name}' started on {args.ip}")
    try:
        while True:
            arm_state = gello_node._build_arm_state()
            gripper_state = gello_node._build_gripper_state()
            gello_node._arm_state_pub.publish(arm_state)
            gello_node._gripper_state_pub.publish(gripper_state)
            pyzlc.sleep(gello_node._state_dt_s)
    except Exception as exc:  # pragma: no cover - runtime feedback only
        pyzlc.error("Publisher stopped due to error: %s", exc)
        return 1
    finally:
        pyzlc.info("Gello publisher exiting")
        return 0


if __name__ == "__main__":
    main()
