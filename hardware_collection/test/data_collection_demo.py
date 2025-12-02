import struct
import shutil
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Optional, NamedTuple, List

import cv2
import hydra
import numpy as np
import torch
import zmq
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
    """Subscribe to the state publisher from RobotControlProxy and decode frames."""

    STATE_BYTES = 636  # expected payload length from the provided C++ encoder

    def __init__(
        self,
        name: str,
        state_pub_addr: str = "tcp://127.0.0.1:5555",
        recv_timeout_ms: int = 1000,
    ):
        self.name = name
        self.state_pub_addr = state_pub_addr
        self.recv_timeout_ms = recv_timeout_ms
        self.ctx = zmq.Context.instance()
        self.socket: Optional[zmq.Socket] = None

    def connect(self) -> None:
        if self.socket is not None:
            return

        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        sock.connect(self.state_pub_addr)
        self.socket = sock
        print(f"Connected to robot state publisher at {self.state_pub_addr}")

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None

    def _read_doubles(self, payload: memoryview, offset: int, count: int):
        end = offset + count * 8
        values = struct.unpack_from(f"<{count}d", payload, offset)
        return torch.tensor(values, dtype=torch.float64), end
    def _decode_state(self, payload: bytes) -> RobotState:
        STRUCT = struct.Struct("!I16d16d7d7d7d7d7d6d6d")
        if len(payload) != self.STATE_BYTES:
            raise ValueError(f"Expected {self.STATE_BYTES} bytes, got {len(payload)}")
        # print("payload len:",len(payload))
        # print("payload:",payload)
        unpacked = STRUCT.unpack(payload)

        offset = 0

        timestamp_ms = unpacked[offset]
        offset += 1

        O_T_EE = torch.tensor(unpacked[offset : offset + 16])
        O_T_EE_d = torch.tensor(unpacked[offset + 16 : offset + 32])
        q = torch.tensor(unpacked[offset + 32 : offset + 39])
        q_d = torch.tensor(unpacked[offset + 39 : offset + 46])
        dq = torch.tensor(unpacked[offset + 46 : offset + 53])
        dq_d = torch.tensor(unpacked[offset + 53 : offset + 60])
        tau_ext_hat_filtered = torch.tensor(unpacked[offset + 60 : offset + 67])
        O_F_ext_hat_K = torch.tensor(unpacked[offset + 67 : offset + 73])
        K_F_ext_hat_K = torch.tensor(unpacked[offset + 73 : offset + 79])

        return RobotState(
            timestamp_ms=timestamp_ms,
            O_T_EE=O_T_EE,
            O_T_EE_d=O_T_EE_d,
            q=q,
            q_d=q_d,
            dq=dq,
            dq_d=dq_d,
            tau_ext_hat_filtered=tau_ext_hat_filtered,
            O_F_ext_hat_K=O_F_ext_hat_K,
            K_F_ext_hat_K=K_F_ext_hat_K,
        )

    def receive_state(self) -> Optional[RobotState]:
        if self.socket is None:
            raise RuntimeError("Robot socket is not connected. Call connect() first.")
        try:
            # Publisher sends a single-frame message at a fixed rate; use recv for clarity
            payload = self.socket.recv()
        except zmq.error.Again:
            return None
        # print("Received payload of length:", len(payload))
        return self._decode_state(payload)


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
    """Subscribe to CameraFrame data from a ZeroMQ PUB publisher."""

    def __init__(self, name: str, address: str, recv_timeout_ms: int = 1000) -> None:
        self.name = name
        self.address = address
        self.recv_timeout_ms = recv_timeout_ms
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.socket.connect(address)

    def close(self) -> None:
        self.socket.close(0)

    def receive_latest_frame(self) -> Optional[CameraFrame]:
        frame: Optional[CameraFrame] = None

        while True:
            try:
                parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as exc:
                print(f"Camera '{self.name}' socket error: {exc}")
                break

            if len(parts) != 2:
                print(f"Camera '{self.name}' sent unexpected multipart message with {len(parts)} parts")
                continue

            header_bytes, image_bytes = parts
            try:
                header = CameraHeader.from_bytes(header_bytes)
            except struct.error as exc:
                print(f"Camera '{self.name}' header decode error: {exc}")
                continue

            image_flat = np.frombuffer(image_bytes, dtype=np.uint8)
            expected = header.width * header.height * header.channels
            if image_flat.size != expected:
                print(
                    f"Camera '{self.name}' frame size mismatch "
                    f"(expected {expected} bytes, got {image_flat.size})"
                )
                continue

            image = image_flat.reshape((header.height, header.width, header.channels)).copy()
            frame = CameraFrame(header=header, image_data=image)

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
        
        capture_interval: float = 0.0,
    ):
        self.robot = robot
        self.robot.connect()

        self.camera_streams: List[RemoteCameraStream] = camera_streams or []
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

        self._writer_queue: Queue = Queue(maxsize=4096)
        self._writer_stop = threading.Event()
        self._writer_thread = threading.Thread(target=self.__writer_loop, daemon=True)
        self._writer_thread.start()

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
                            self._writer_stop.set()
                            self._writer_queue.join()
                            self._writer_thread.join(timeout=2.0)
                            print("Writer thread stopped.")
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
                self._writer_queue.put_nowait(
                    (frame_path, header_path, image_bgr, frame.header.to_bytes(), self.camera_names[idx], self.cur_timestep)
                )
            except Full:
                print(f"Camera '{self.camera_names[idx]}' writer queue full, dropping frame {self.cur_timestep}")
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
        self._writer_queue.join()

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

    def __writer_loop(self) -> None:
        """Background worker to write images and headers without blocking capture."""
        while not self._writer_stop.is_set() or not self._writer_queue.empty():
            try:
                frame_path, header_path, image_bgr, header_bytes, cam_name, step = self._writer_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                cv2.imwrite(str(frame_path), image_bgr)
                # with open(header_path, "wb") as header_file:
                    # header_file.write(header_bytes)
            except Exception as exc:
                print(f"Failed to write frame {frame_path} (cam {cam_name}, step {step}): {exc}")
            finally:
                self._writer_queue.task_done()

    def __close_hardware_connections(self):
        self.robot.close()

        for camera in self.camera_streams:
            try:
                camera.close()
            except Exception as exc:
                print(f"Failed to close camera stream {camera.name}: {exc}")

        self._writer_stop.set()
        self._writer_thread.join(timeout=2.0)


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig):

    camera_cfg = cfg.get("camera_streams", cfg.get("cameras", {}))
    camera_streams: List[RemoteCameraStream] = []
    if camera_cfg:
        for name, cam_cfg in camera_cfg.items():
            address = cam_cfg.get("address")
            if not address:
                print(f"Camera '{name}' is missing an 'address' entry in the config, skipping.")
                continue
            recv_timeout = cam_cfg.get("recv_timeout_ms", 1000)
            camera_streams.append(
                RemoteCameraStream(
                    name=name,
                    address=address,
                    recv_timeout_ms=recv_timeout,
                )
            )

    state_pub_addr = cfg.get("state_pub_addr", "tcp://127.0.0.1:5555")
    robot_name = cfg.get("robot_name", "franka")
    robot = Robot(name=robot_name, state_pub_addr=state_pub_addr)
    robot.connect()
    data_collection_manager = DataCollectionManager(
        robot=robot,
        data_dir=Path(cfg.data_dir),
        camera_streams=camera_streams,
        capture_interval=cfg.get("capture_interval", 0.001),
    )
    data_collection_manager.start_key_listener()

if __name__ == "__main__":
    main()
