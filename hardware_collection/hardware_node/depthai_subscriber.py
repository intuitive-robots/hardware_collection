import os
import time
import pyzlc
from ..camera.camera import CameraFrame

def image_subscriber_node(topic: str, output_dir: str = "received_images"):
    pyzlc.init("SubscriberNode", "10.172.218.210")
    
    """Subscribe to a pyzlc topic and save received images to disk."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Subscribing to topic '{topic}' and saving images to '{output_dir}'...")

    def handle_image(msg):
        try:
            frame = CameraFrame.from_bytes(msg)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            ms = int((time.time() % 1) * 1000)
            fname = os.path.join(output_dir, f"img_{ts}_{ms:03d}.png")
            frame.save_image(fname)
            print(f"Saved image: {fname}")
        except Exception as e:
            print(f"Failed to decode or save image: {e}")

    pyzlc.register_subscriber_handler(topic, handle_image)
    print("Waiting for images... Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting subscriber node.")
    pyzlc.spin()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DepthAI Image Subscriber Node via pyzlc")
    parser.add_argument('--topic', type=str, required=True, help='pyzlc topic to subscribe to')
    parser.add_argument('--output', type=str, default='received_images', help='Directory to save received images')
    args = parser.parse_args()
    image_subscriber_node(args.topic, args.output)
