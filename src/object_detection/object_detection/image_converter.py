import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
mode = "detect"
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
names = model.names
print(names)


class ImageConverter(Node):
    def __init__(self):
        super().__init__("image_converter")
        self.subscription = self.create_subscription(
            Image, "/theta_z1/rgb", self.listener_callback, 10
        )
        self.publisher = self.create_publisher(Image, "converted_image", 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Perform inference
        results = model(cv_image)

        boxes = results[0].boxes
        if mode == "detect":
            for box in boxes:
                confidence_level = float(box.conf.cpu().numpy())
                object_name = names[int(box.cls.cpu().numpy())]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                print(f"{x1} {y1} {x2} {y2}")
                print(box)

                # Choose color based on class
                color = GREEN if int(box.cls.cpu().numpy()) == 0 else BLUE

                # Draw the bounding box on the image
                if int(box.cls.cpu().numpy()) == 0:
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 1)
                x1, y1, x2, y2 = map(int, box.xywh[0].cpu().numpy())
                image = cv2.line(image, x2, y2, BLUE, thickness)

        # quit()
        converted_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.publisher.publish(converted_msg)


def main(args=None):
    rclpy.init(args=args)
    image_converter = ImageConverter()
    rclpy.spin(image_converter)
    image_converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
