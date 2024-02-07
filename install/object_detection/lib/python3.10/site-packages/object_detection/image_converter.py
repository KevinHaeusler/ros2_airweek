import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs.msg
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import supervision as sv
from ultralytics import YOLO
import colorsys
import math

facemodel = YOLO(
    "/home/kevin/ros2_airweek/src/object_detection/object_detection/yolov8n-face.pt"
)
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


def calculate_marker_position(x, z, image_width, max_radial_distance):
    # Convert x to an angle (0 to 360 degrees, then to radians)
    angle_rad = (x / image_width) * 2 * math.pi

    # Use z value as radial distance, may scale if needed
    radial_distance = z * max_radial_distance  # Scale z to adjust distance from center

    # Convert polar coordinates (angle, radial_distance) to Cartesian coordinates
    cartesian_x = radial_distance * math.cos(angle_rad)
    cartesian_y = radial_distance * math.sin(angle_rad)

    return cartesian_x, cartesian_y


def convert_to_circular_layout(x, y, image_width, image_height):
    # Calculate the horizontal angle (theta) from the x-coordinate
    theta = (x / image_width) * 2 * math.pi  # Convert pixel x to radians

    # Map y to a radius or another metric as needed
    # This example simply maps y linearly; adjust based on your visualization needs
    radius = y  # Placeholder for radius calculation

    # Convert polar coordinates (theta, radius) to Cartesian for visualization
    cartesian_x = radius * math.cos(theta)
    cartesian_y = radius * math.sin(theta)

    return cartesian_x, cartesian_y


def adjust_for_rviz(cartesian_x, cartesian_y, scale_factor=10):
    # Scale up for visibility in RViz2
    adjusted_x = cartesian_x * scale_factor
    adjusted_y = cartesian_y * scale_factor
    return adjusted_x, adjusted_y


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
        self.image_publisher = self.create_publisher(Image, "converted_image", 10)
        self.marker_publisher = self.create_publisher(
            MarkerArray, "skeleton_markers", 10
        )  # MarkerArray publisher
        self.bridge = CvBridge()
        self.color_index = 0
        self.person_colors = generate_distinct_colors(10)

    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        person_index = 0  # Index to track which person we're on
        max_radial_distance = 5
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
            # object_distance = names[int(box.cls.cpu().numpy())]

            # Constants for distance and height
            distance = 4.0
            max_height = 8.5

            # Create MarkerArray
            marker_array = MarkerArray()

            # Iterate over skeletons in chunks of 17
            for i in range(0, len(skeleton), 17):
                keypoints = skeleton[i : i + 17]

                # Find minimum and maximum y-coordinate
                min_y = min(y for x, y in keypoints)
                max_y = max(y for x, y in keypoints)

                for idx, (x, y) in enumerate(keypoints):
                    # Normalize y-coordinate to range [0, max_height]
                    normalized_y = (
                        ((y - min_y) / (max_y - min_y)) * max_height
                        if max_y > min_y
                        else 0
                    )

                    # Create marker with adjusted positions
                    marker = Marker()
                    marker.header.frame_id = "camera_link"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.id = i + idx
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = (
                        float(x) / 320 * distance
                    )  # Normalize x-coordinate to the range [-distance, distance]
                    marker.pose.position.y = distance
                    marker.pose.position.z = (
                        normalized_y  # Use normalized y-coordinate as z-coordinate
                    )
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0  # Don't forget to set alpha!
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0

                    marker_array.markers.append(marker)
        converted_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_publisher.publish(converted_msg)
        self.marker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    image_converter = ImageConverter()
    rclpy.spin(image_converter)
    image_converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
