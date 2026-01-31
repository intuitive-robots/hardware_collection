import pyzlc
import cv2
import time
from hardware_collection.camera.camera_depthai import DepthAICamera

if __name__ == "__main__":
    pyzlc.init("test_depthai_camera", "192.168.0.134")
    cameras = DepthAICamera.get_devices(amount=1, height=480, width=640)
    if not cameras:
        pyzlc.error("No DepthAI cameras found")
        exit(1)
    
    cam = cameras[0]
    pyzlc.info("Initializing DepthAI camera (device_id=%s)", cam.device_id)
    while True:
        frame = cam.capture_image()
        # bytes to image display
        image = frame["image_data"]
        start = time.time.perf_counter()
        cv2.imshow("DepthAI Camera", image)
        print("Display time:", (time.time() - start) * 1000, "ms")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()