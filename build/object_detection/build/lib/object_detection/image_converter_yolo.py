import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import colorsys


facemodel = YOLO("yolov8n-face.pt")
model = YOLO("yolov8n-pose.pt")


def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5
        saturation = 1.0
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class ImageConverter(Node):
    def __init__(self):
        super().__init__("image_converter")
        self.subscription = self.create_subscription(
            Image, "/theta_z1/rgb", self.listener_callback, 10
        )
        self.image_publisher = self.create_publisher(Image, "converted_image", 10)
        self.bridge = CvBridge()
        self.color_index = 0
        self.person_colors = generate_distinct_colors(10)

    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        results = model(cv_image)
        faceresults = facemodel(cv_image)
        keypoints = results[0].keypoints.xy
        skeleton = []
        skeleton_depth = []
        skeleton_maps = (
            (5, 6),  # s Shoulders
            (5, 7),
            (7, 9),  # Left arm
            (6, 8),
            (8, 10),  # Right arm
            (5, 11),
            (6, 12),  # Torso
            (11, 13),
            (13, 15),  # Left leg
            (12, 14),
            (14, 16),  # Right leg
        )
        for keypoint_set in keypoints:
            # Iterating over each keypoint in the set
            for keypoint in keypoint_set:
                x, y = int(keypoint[0].cpu().numpy()), int(keypoint[1].cpu().numpy())
                if x != 0 and y != 0:
                    cv2.circle(cv_image, (x, y), 1, BLUE, -1)

                skl = (x, y)
                skeleton.append(skl)
                if len(skeleton) % 17 == 0:
                    for skeleton_map in skeleton_maps:

                        cv2.line(
                            cv_image,
                            tuple(skeleton[skeleton_map[0]]),
                            tuple(skeleton[skeleton_map[1]]),
                            BLUE,
                            1,
                        )
        keypoints = results[0].keypoints.xyn
        for keypoint_set in keypoints:
            # Iterating over each keypoint in the set
            for keypoint in keypoint_set:
                x, y = float(keypoint[0].cpu().numpy()), float(
                    keypoint[1].cpu().numpy()
                )
                skl = (x, y)
                skeleton_depth.append(skl)

        boxes = faceresults[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1231, y123123, w, h = map(int, box.xywh[0].cpu().numpy())

            k = 112
            distance_label = f"{(k/(w*h)) ** 0.5:.2f}"

            # Draw the bounding box on the image
            if int(box.cls.cpu().numpy()) == 0:
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), GREEN, 1)
                cv2.putText(
                    cv_image,
                    distance_label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    GREEN,
                    1,
                )

        converted_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_publisher.publish(converted_msg)


def main(args=None):
    rclpy.init(args=args)
    image_converter = ImageConverter()
    rclpy.spin(image_converter)
    image_converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
