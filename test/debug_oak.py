import depthai as dai
import cv2

#
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(640, 480)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)


xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
cam.preview.link(xout.input)

# 
with dai.Device(pipeline) as device:
    print("USB speed:", device.getUsbSpeed())
    q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    while True:
        frame = q.get().getCvFrame()
        #print("frame:", frame.shape, "min:", frame.min(), "max:", frame.max())
        cv2.imshow("OAK-D-Lite Color (CAM_A IMX214)", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()