import sys
import os
import time

import numpy as np
import pyzlc

# Add hardware directory to path to import GelloAgent
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hardware.gello_zlc import GelloAgent

from franka_control_client.franka_robot.franka_arm import (
    ControlMode,
    RemoteFranka,
)
from franka_control_client.franka_robot.franka_gripper import (
    RemoteFrankaGripper,
)

if __name__ == "__main__":
    # Initialize Gello agent
    gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EVRT-if00-port0"
    gello = GelloAgent(port=gello_port)
    
    pyzlc.init("gello_control_client", "127.0.0.2")
    robot = RemoteFranka("FrankaPanda")
    # gripper = RemoteFrankaGripper("192.168.0.", 5557)
    robot.connect()
    # gripper.start_gripper_control()
    print(robot.get_franka_arm_control_mode())
    print(robot.get_franka_arm_state())
    gello_joint_state = gello._robot.get_joint_state()
    arm_joints = np.array(gello_joint_state[:-1], dtype=np.float64)
    print("Aligning Franka to Gello joints:", arm_joints.round(3).tolist())
    robot.move_franka_arm_to_joint_position(tuple(arm_joints))
    robot.set_franka_arm_control_mode(ControlMode.HybridJointImpedance)
    try:
        while True:
            # Get Gello joint state (8 joints: 7 arm + 1 gripper)
            gello_joint_state = gello._robot.get_joint_state()
            
            # Extract arm joints (first 7) and gripper (last joint)
            arm_joints = np.array(gello_joint_state[:-1], dtype=np.float64)
            gripper_value = float(gello_joint_state[-1])
            
            # Send arm joint positions to robot
            robot.send_joint_position_command(arm_joints)
            
            # print(f"Gello joints: {arm_joints.round(3).tolist()} | Gripper: {gripper_value:.3f}")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    for _ in range(100):
        time.sleep(0.1)
