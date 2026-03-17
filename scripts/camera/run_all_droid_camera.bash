#!/bin/bash

# Kill all children on Ctrl+C
trap "echo 'Stopping all...'; kill 0" SIGINT

python hardware_collection/hardware_node/zed_cam_node.py --config configs/zed_config/102_zed_left_cam.yaml &
python hardware_collection/hardware_node/zed_cam_node.py --config configs/zed_config/102_zed_right_cam.yaml &
python hardware_collection/hardware_node/zed_cam_node.py --config configs/zed_config/102_zed_wrist_cam.yaml &

wait