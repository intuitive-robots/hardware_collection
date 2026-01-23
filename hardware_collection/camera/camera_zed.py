from __future__ import annotations
import time
import cv2
import numpy as np
import pyudev

from .camera import AbstractCamera, CameraFrame, CameraHeader


class ZEDCamera(AbstractCamera):

    def __init__(self, device_path, publish_topic: str):
        self.device_path = device_path
        super().__init__(publish_topic=publish_topic)
        self.initialize()

    def initialize(self):
        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open: {self.device_path}")

    def capture_image(self) -> CameraFrame:
        """Capture one frame of RGB image from ZED camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("cannot read frame from ZED camera")

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        header = CameraHeader(
            width=frame_rgb.shape[1],
            height=frame_rgb.shape[0],
            channels=frame_rgb.shape[2],
            timestamp=time.time(),
            frame_id=0,
        )

        return CameraFrame(header=header, image_data=frame_rgb)

    @staticmethod
    def get_devices(
        amount: int = -1,
        topics: list[str] | None = None,
    ) -> list["ZEDCamera"]:
        """
        Automatically find ZED cameras using pyudev and return a list of ZEDCamera instances.
        """

        context = pyudev.Context()
        zed_nodes = []

        for dev in context.list_devices(subsystem="video4linux"):
            parent = dev.find_parent("usb", "usb_device")
            if parent is None:
                continue

            if parent.get("ID_VENDOR_ID") == "2b03":  # ZED Vendor ID
                zed_nodes.append(dev.device_node)

        if not zed_nodes:
            return []

        cams = []
        counter = 0

        for devnode in zed_nodes:
            if amount != -1 and counter >= amount:
                break
            topic = topics[counter] if topics and counter < len(topics) else f"zed_{counter}"
            cams.append(ZEDCamera(device_path=devnode, publish_topic=topic))
            counter += 1

        return cams

if __name__ == "__main__":
    cams = ZEDCamera.get_devices()
    for cam in cams:
        cam.publish_image(cam.capture_image())
