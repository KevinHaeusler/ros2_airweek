import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import supervision as sv
from ultralytics import YOLO

facemodel = YOLO(
    "/home/kevin/ros2_airweek/src/object_detection/object_detection/yolov8n-face.pt"
)

model = YOLO("yolov8m-pose.pt")

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
        faceresults = facemodel(cv_image)

        keypoints = results[0].keypoints
        boxes = faceresults[0].boxes
        # for keypoint_set in keypoints:
        # pose keypoints
        #     print(results[0].keypoints)
        #     for keypoint in keypoint_set:
        #         x, y = map(int, keypoint.cpu().numpy())
        # Draw keypoints
        #         cv2.circle(cv_image, (x, y), 3, GREEN, -1)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            color = GREEN if int(box.cls.cpu().numpy()) == 0 else BLUE

            # Draw the bounding box on the image
            if int(box.cls.cpu().numpy()) == 0:
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 1)

            object_distance = names[int(box.cls.cpu().numpy())]
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
