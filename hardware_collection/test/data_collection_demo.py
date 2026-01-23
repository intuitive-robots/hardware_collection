import shutil
import time
import os
from datetime import datetime
from pathlib import Path
from queue import Full #used to handle full queue exception
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional, NamedTuple, List

import cv2
import hydra
import msgpack
import numpy as np
import pyzlc
import torch
from omegaconf import DictConfig

from hardware_collection.camera.camera import CameraFrame, CameraHeader
from utils.keyboard_input import NonBlockingKeyPress


class RobotState(NamedTuple):
    timestamp_ms: int
    O_T_EE: torch.Tensor  # (16,) flattened
    O_T_EE_d: torch.Tensor  # (16,) flattened
    q: torch.Tensor  # (7,)
    q_d: torch.Tensor  # (7,)
    dq: torch.Tensor  # (7,)
    dq_d: torch.Tensor  # (7,)
    tau_ext_hat_filtered: torch.Tensor  # (7,)
    O_F_ext_hat_K: torch.Tensor  # (6,)
    K_F_ext_hat_K: torch.Tensor  # (6,)


class Robot:
    """Subscribe to robot state via ZeroLanCom."""

    def __init__(
        self,
        name: str,
        state_topic: Optional[str] = None,
        recv_timeout_ms: int = 1000,
    ):
        self.name = name
        # FrankaControlProxy publishes on "<name>/franka_arm_state"
        self.state_topic = state_topic or f"{name}/franka_arm_state"
        self.recv_timeout_ms = recv_timeout_ms
        self.latest_state: Optional[bytes | dict] = None

    def connect(self) -> None:
        """Register subscriber for robot state."""
        try:
            pyzlc.register_subscriber_handler(
                self.state_topic, self._handle_state
            )
            print(f"Connected to robot state topic '{self.state_topic}' via ZeroLanCom")
        except Exception as e:
            print(f"Warning: Failed to connect to robot state: {e}")

    def _handle_state(self, msg) -> None:
        """Handle incoming state data (dict from msgpack or raw bytes)."""
        try:
            if isinstance(msg, (bytes, bytearray)):
                self.latest_state = bytes(msg)
            else:
                self.latest_state = msg
        except Exception as e:
            print(f"Failed to process robot state: {e}")

    def close(self) -> None:
        """Close connection."""
        self.latest_state = None

    def _decode_state(self, payload) -> RobotState:
        """
        Decode FrankaControlProxy FrankaArmState published via pyzlc msgpack.
        Expected keys: time_ms, O_T_EE, O_T_EE_d, q, q_d, dq, dq_d,
        tau_ext_hat_filtered, O_F_ext_hat_K, K_F_ext_hat_K.
        """
        if isinstance(payload, (bytes, bytearray)):
            data = msgpack.unpackb(payload, raw=False)
        elif isinstance(payload, dict):
            data = payload
        else:
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        required_keys = [
            "time_ms",
            "O_T_EE",
            "O_T_EE_d",
            "q",
            "q_d",
            "dq",
            "dq_d",
            "tau_ext_hat_filtered",
            "O_F_ext_hat_K",
            "K_F_ext_hat_K",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing key '{key}' in robot state message")

        to_tensor = lambda x: torch.tensor(x, dtype=torch.float64)

        return RobotState(
            timestamp_ms=int(data["time_ms"]),
            O_T_EE=to_tensor(data["O_T_EE"]),
            O_T_EE_d=to_tensor(data["O_T_EE_d"]),
            q=to_tensor(data["q"]),
            q_d=to_tensor(data["q_d"]),
            dq=to_tensor(data["dq"]),
            dq_d=to_tensor(data["dq_d"]),
            tau_ext_hat_filtered=to_tensor(data["tau_ext_hat_filtered"]),
            O_F_ext_hat_K=to_tensor(data["O_F_ext_hat_K"]),
            K_F_ext_hat_K=to_tensor(data["K_F_ext_hat_K"]),
        )

    def receive_state(self) -> Optional[RobotState]:
        """Receive latest robot state if available."""
        if self.latest_state is None:
            return None
        
        try:
            return self._decode_state(self.latest_state)
        except Exception as e:
            print(f"Failed to decode robot state: {e}")
            return None


class CollectionData:
    def __init__(self):
        self.timestamp_ms_list = []
        self.O_T_EE_list = []
        self.O_T_EE_d_list = []
        self.q_list = []
        self.q_d_list = []
        self.dq_list = []
        self.dq_d_list = []
        self.tau_ext_hat_filtered_list = []

    def append(self, state: RobotState):
        self.timestamp_ms_list.append(state.timestamp_ms)
        self.O_T_EE_list.append(state.O_T_EE)
        self.O_T_EE_d_list.append(state.O_T_EE_d)
        self.q_list.append(state.q)
        self.q_d_list.append(state.q_d)
        self.dq_list.append(state.dq)
        self.dq_d_list.append(state.dq_d)
        self.tau_ext_hat_filtered_list.append(state.tau_ext_hat_filtered)

    def save(self, path: Path):

        tensor_lists = [
            torch.tensor(self.timestamp_ms_list, dtype=torch.int64),
            torch.stack(self.O_T_EE_list),
            torch.stack(self.O_T_EE_d_list),
            torch.stack(self.q_list),
            torch.stack(self.q_d_list),
            torch.stack(self.dq_list),
            torch.stack(self.dq_d_list),
            torch.stack(self.tau_ext_hat_filtered_list),
        ]
        paths = [
            path / "timestamp_ms.pt",
            path / "O_T_EE.pt",
            path / "O_T_EE_d.pt",
            path / "q.pt",
            path / "q_d.pt",
            path / "dq.pt",
            path / "dq_d.pt",
            path / "tau_ext_hat_filtered.pt",
        ]

        for d, p in zip(tensor_lists, paths):

            if d.numel() == 0:
                print(f"Skip saving '{p}' since it is empty")
                continue

            torch.save(d, p)
            print(f"Successfully saved '{p}'")


class RemoteCameraStream:
    """Subscribe to camera frames via ZeroLanCom."""

    def __init__(self, name: str, topic: str, recv_timeout_ms: int = 1000) -> None:
        self.name = name
        self.topic = topic
        self.recv_timeout_ms = recv_timeout_ms
        self.latest_frame: Optional[CameraFrame] = None

    def connect(self) -> None:
        """Register subscriber for camera frames."""
        try:
            pyzlc.register_subscriber_handler(
                self.topic, self._handle_frame
            )
            print(f"Connected to camera stream '{self.name}' on topic '{self.topic}'")
        except Exception as e:
            print(f"Warning: Failed to connect to camera stream '{self.name}': {e}")

    def _handle_frame(self, msg: bytes) -> None:
        """Handle incoming frame data."""
        try:
            self.latest_frame = CameraFrame.from_bytes(msg)
        except Exception as e:
            print(f"Camera '{self.name}' frame decode error: {e}")

    def close(self) -> None:
        """Close the subscriber connection."""
        pass

    def receive_latest_frame(self) -> Optional[CameraFrame]:
        """Get the latest frame if available."""
        if self.latest_frame is None:
            return None
        
        frame = self.latest_frame
        self.latest_frame = None  # Clear after retrieval
        return frame


class DataCollectionManager:
    """
    Collect data from a single robot subscriber and optional sensors (e.g. cameras).
    Interactive workflow: press 'n' to start collection, 's' to save, 'd' to discard, 'q' to exit.
    """

    def __init__(
        self,
        robot: Robot,
        data_dir: Path,
        camera_streams: Optional[List[RemoteCameraStream]] = None,
        writer_pool_max_workers: Optional[int] = None,
        writer_max_pending_writes: int = 4096,
        capture_interval: float = 0.0,
    ):
        self.robot = robot
        self.robot.connect()

        self.camera_streams: List[RemoteCameraStream] = camera_streams or []
        
        # Connect all camera streams
        for camera in self.camera_streams:
            camera.connect()
        
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

        self._max_pending_writes = int(writer_max_pending_writes)
        if writer_pool_max_workers is None:
            max_workers = max(2, min(8, (os.cpu_count() or 4)))
        else:
            max_workers = max(1, int(writer_pool_max_workers))
        self._writer_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._writer_futures = []

        self.camera_dirs: List[Path] = []
        self.camera_names: List[str] = []
        self.camera_timestamps: List[list[float]] = []
        self.camera_frame_idx = [0] * len(self.camera_streams)
        self.timestamps = []
        self.cur_timestep = 0
        self.capture_interval = capture_interval

    def start_key_listener(self):
        print("📦 Press 'n' to collect new data or 'q' to quit data collection")

        with NonBlockingKeyPress() as kp:
            quit = False
            while not quit:

                # Update input
                key = kp.get_data()
                if key == "q":
                    quit = True
                elif key == "n":
                    quit = False

                if key == "n":
                    print("📯 Preparing for new data collection")

                    self.__create_new_recording_dir()
                    self.__create_empty_data()

                    print("🚀 Start! Press 's' to save collected data or 'd' to discard.")

                    self.timestamps = []
                    self.cur_timestep = 0
                    collect = True
                    while collect:

                        # Update input
                        key = kp.get_data()
                        if key in ["s", "d"]:
                            collect = False

                        self.__collection_step()

                    else:
                        if key == "s":
                            print("Saving data ...")

                            self.__save_data()

                            print("Saved!")
                        elif key == "d":
                            print("Discarding data ...")
                            self.__flush_writes()
                            shutil.rmtree(self.record_dir)

                            print("Discarded!")

                        print(
                            "📦 Press 'n' to collect new data or 'q' to quit data collection"
                        )

        print("⌛ Ending data collection...")
        self.__close_hardware_connections()

    def __create_new_recording_dir(self):
        self.record_dir = self.data_dir / datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.record_dir.mkdir()

        self.robot_dir = self.record_dir / self.robot.name
        self.robot_dir.mkdir()

        self.sensors_dir = self.record_dir / "sensors"
        self.sensors_dir.mkdir()

        self.camera_dirs = []
        self.camera_names = []
        for idx, camera in enumerate(self.camera_streams):
            cam_name = self.__camera_identifier(camera, idx)
            device_dir = self.sensors_dir / cam_name
            device_dir.mkdir()
            self.camera_dirs.append(device_dir)
            self.camera_names.append(cam_name)

    def __create_empty_data(self):
        self.robot_data = CollectionData()
        self.camera_timestamps = [[] for _ in self.camera_streams]

    def __camera_identifier(self, camera: RemoteCameraStream, idx: int) -> str:
        candidate = camera.name or camera.address or f"camera_{idx}"
        sanitized = (
            str(candidate)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        return sanitized or f"camera_{idx}"

    def __capture_camera_frames(self) -> None:
        if not self.camera_streams:
            return
        
        for idx, stream in enumerate(self.camera_streams):
            camera_dir = self.camera_dirs[idx]
            frame = stream.receive_latest_frame()
            if frame is None:
                continue
            frame_idx = self.camera_frame_idx[idx]
            self.camera_frame_idx[idx] += 1

            image_rgb = frame.image_data
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            frame_path = camera_dir / f"{frame_idx:06d}.png"
            header_path = frame_path.with_suffix(".header")

            try:
                self.__submit_frame_write(
                    frame_path,
                    header_path,
                    image_bgr,
                    frame.header.to_bytes(),
                    self.camera_names[idx],
                    self.cur_timestep,
                )
            except Full:
                print(
                    f"Camera '{self.camera_names[idx]}' writer pool backlog full, dropping frame {self.cur_timestep}"
                )
                continue

            self.camera_timestamps[idx].append(frame.header.timestamp)

    def __collection_step(self):
        state = self.robot.receive_state()
        # print("Received robot state:", state)
        if state is None:
            return

        cur_time = time.time()  # seconds since epoch for spacing samples

        if not self.timestamps or cur_time - self.timestamps[-1] >= self.capture_interval:
            self.robot_data.append(state)
            self.timestamps.append(cur_time)

            self.__capture_camera_frames()

            self.cur_timestep += 1

    def __save_data(self):

        # Ensure all enqueued frames are written before finalizing save
        self.__flush_writes()

        timestamps_path = self.record_dir / "timestamps.pt"
        torch.save(torch.tensor(self.timestamps, dtype=torch.float64), timestamps_path)
        print(f"Successfully saved '{timestamps_path}'")
        self.robot_data.save(self.robot_dir)

        self.__save_camera_metadata()
        self.__report_camera_rates()

        # determine average frame rate from timestamps
        if len(self.timestamps) > 1:
            print(f"Robot states frame rate: {len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0]):.2f} Hz")
        else:
            print("Robot states frame rate: only one sample captured")

    def __save_camera_metadata(self) -> None:
        for camera_dir, timestamps, name in zip(
            self.camera_dirs, self.camera_timestamps, self.camera_names
        ):
            if not timestamps:
                print(f"No frames captured for camera '{name}', skipping timestamp save.")
                continue

            # ts_path = camera_dir / "timestamps.pt"
            # torch.save(torch.tensor(timestamps, dtype=torch.float64), ts_path)
            # print(f"Successfully saved '{ts_path}'")

    def __report_camera_rates(self) -> None:
        """Report average frame rate for each camera based on captured timestamps."""
        for name, timestamps in zip(self.camera_names, self.camera_timestamps):
            if len(timestamps) <= 1:
                print(f"Camera '{name}' frame rate: insufficient samples ({len(timestamps)})")
                continue

            span = timestamps[-1] - timestamps[0]
            if span <= 0:
                print(f"Camera '{name}' frame rate: invalid timestamp span")
                continue

            fps = len(timestamps) / span
            print(f"Camera '{name}' frame rate: {fps:.2f} Hz")

    def __submit_frame_write(
        self,
        frame_path: Path,
        header_path: Path,
        image_bgr: np.ndarray,
        header_bytes: bytes,
        cam_name: str,
        step: int,
    ) -> None:
        self.__prune_completed_writes()
        if len(self._writer_futures) >= self._max_pending_writes:
            raise Full

        future = self._writer_pool.submit(
            self.__write_frame, frame_path, header_path, image_bgr, header_bytes, cam_name, step
        )
        self._writer_futures.append(future)

    def __prune_completed_writes(self) -> None:
        if not self._writer_futures:
            return
        self._writer_futures = [f for f in self._writer_futures if not f.done()]

    def __flush_writes(self) -> None:
        if not self._writer_futures:
            return
        wait(self._writer_futures)
        self.__prune_completed_writes()

    @staticmethod
    def __write_frame(
        frame_path: Path,
        header_path: Path,
        image_bgr: np.ndarray,
        header_bytes: bytes,
        cam_name: str,
        step: int,
    ) -> None:
        try:
            cv2.imwrite(str(frame_path), image_bgr)
            # with open(header_path, "wb") as header_file:
            #     header_file.write(header_bytes)
        except Exception as exc:
            print(f"Failed to write frame {frame_path} (cam {cam_name}, step {step}): {exc}")

    def __close_hardware_connections(self):
        self.robot.close()

        for camera in self.camera_streams:
            try:
                camera.close()
            except Exception as exc:
                print(f"Failed to close camera stream {camera.name}: {exc}")

        self.__flush_writes()
        self._writer_pool.shutdown(wait=True, cancel_futures=False)


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig):

    camera_cfg = cfg.get("camera_streams", cfg.get("cameras", {}))
    camera_streams: List[RemoteCameraStream] = []
    if camera_cfg:
        for name, cam_cfg in camera_cfg.items():
            topic = cam_cfg.get("topic")
            if not topic:
                print(f"Camera '{name}' is missing a 'topic' entry in the config, skipping.")
                continue
            recv_timeout = cam_cfg.get("recv_timeout_ms", 1000)
            camera_streams.append(
                RemoteCameraStream(
                    name=name,
                    topic=topic,
                    recv_timeout_ms=recv_timeout,
                )
            )

    state_topic = cfg.get("state_topic")
    robot_name = cfg.get("robot_name", "FrankaPanda")
    robot = Robot(name=robot_name, state_topic=state_topic)
    robot.connect()
    data_collection_manager = DataCollectionManager(
        robot=robot,
        data_dir=Path(cfg.data_dir),
        camera_streams=camera_streams,
        writer_pool_max_workers=cfg.get("writer_pool_max_workers"),
        writer_max_pending_writes=cfg.get("writer_max_pending_writes", 4096),
        capture_interval=cfg.get("capture_interval", 0.001),
    )
    data_collection_manager.start_key_listener()

if __name__ == "__main__":
    main()
