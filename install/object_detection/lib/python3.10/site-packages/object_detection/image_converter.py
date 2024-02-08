import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import openpifpaf
import numpy as np


class ImageConverter(Node):
    def __init__(self):
        super().__init__("image_converter")
        self.subscription = self.create_subscription(
            Image, "/theta_z1/rgb", self.listener_callback, 10
        )
        self.image_publisher = self.create_publisher(Image, "converted_image", 10)
        self.bridge = CvBridge()

        # Initialize OpenPifPaf predictor
        self.predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k30")

    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Perform pose estimation with OpenPifPaf
        predictions, gt_anns, image_meta = self.predictor.numpy_image(image_rgb)

        # Draw poses on the image
        for pred in predictions:
            self.draw_pose(cv_image, pred)

        # Publish the OpenCV image as a ROS message
        converted_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_publisher.publish(converted_msg)

    def estimate_distance(self, keypoints):
        # Constants
        REF_ARM_LENGTH_CM = 65  # Reference arm length in centimeters
        REF_HIP_SHOULDER_DISTANCE_CM = (
            60  # Reference hip to shoulder distance in centimeters
        )

        # Keypoint indices for relevant body parts
        left_shoulder_idx, right_shoulder_idx = 5, 6
        left_elbow_idx, right_elbow_idx = 7, 8
        left_hip_idx, right_hip_idx = 11, 12

        # Calculate arm lengths
        left_arm_length_px = np.linalg.norm(
            np.array(keypoints[left_shoulder_idx][:2])
            - np.array(keypoints[left_elbow_idx][:2])
        )
        right_arm_length_px = np.linalg.norm(
            np.array(keypoints[right_shoulder_idx][:2])
            - np.array(keypoints[right_elbow_idx][:2])
        )

        # Calculate hip to shoulder distances
        left_hip_shoulder_distance_px = np.linalg.norm(
            np.array(keypoints[left_shoulder_idx][:2])
            - np.array(keypoints[left_hip_idx][:2])
        )
        right_hip_shoulder_distance_px = np.linalg.norm(
            np.array(keypoints[right_shoulder_idx][:2])
            - np.array(keypoints[right_hip_idx][:2])
        )

        # Calculate ratios of lengths to reference sizes
        left_arm_ratio = left_arm_length_px / REF_ARM_LENGTH_CM
        right_arm_ratio = right_arm_length_px / REF_ARM_LENGTH_CM
        left_hip_shoulder_ratio = (
            left_hip_shoulder_distance_px / REF_HIP_SHOULDER_DISTANCE_CM
        )
        right_hip_shoulder_ratio = (
            right_hip_shoulder_distance_px / REF_HIP_SHOULDER_DISTANCE_CM
        )

        # Use the average of ratios to estimate distance
        avg_ratio = (
            left_arm_ratio
            + right_arm_ratio
            + left_hip_shoulder_ratio
            + right_hip_shoulder_ratio
        ) / 4

        # Assume an average distance for reference sizes
        avg_distance_cm = (REF_ARM_LENGTH_CM + REF_HIP_SHOULDER_DISTANCE_CM) / 2

        # Adjust coefficient to scale down the estimated distance
        coefficient = 0.7  # Adjust as needed
        distance_cm = (avg_distance_cm * coefficient) / avg_ratio

        return distance_cm

    def draw_pose(self, img, pred):
        # Define the color for each limb
        # Define a list of colors for different parts of the body
        LIMB_COLORS = [
            (255, 0, 0),  # Nose to left eye
            (255, 85, 0),  # Nose to right eye
            (255, 170, 0),  # Left eye to left ear
            (255, 255, 0),  # Right eye to right ear
            (170, 255, 0),  # Nose to left shoulder
            (85, 255, 0),  # Nose to right shoulder
            (0, 255, 0),  # Left shoulder to left elbow
            (0, 255, 85),  # Right shoulder to right elbow
            (0, 255, 170),  # Left elbow to left wrist
            (0, 255, 255),  # Right elbow to right wrist
            (0, 170, 255),  # Left shoulder to right shoulder
            (0, 85, 255),  # Left shoulder to left hip
            (0, 0, 255),  # Right shoulder to right hip
            (85, 0, 255),  # Left hip to right hip
            (170, 0, 255),  # Left hip to left knee
            (255, 0, 255),  # Right hip to right knee
            (255, 0, 170),  # Left knee to left ankle
            (255, 0, 85),  # Right knee to right ankle
        ]

        # Each tuple is (start_index, end_index, color_index)
        LIMB_CONNECTIONS = [
            (0, 1, 0),
            (0, 2, 1),
            (1, 3, 2),
            (2, 4, 3),
            (0, 5, 4),
            (0, 6, 5),
            (5, 7, 6),
            (6, 8, 7),
            (7, 9, 8),
            (8, 10, 9),
            (5, 6, 10),
            (5, 11, 11),
            (6, 12, 12),
            (11, 12, 13),
            (11, 13, 14),
            (12, 14, 15),
            (13, 15, 16),
            (14, 16, 17),
        ]

        keypoints = pred.data
        # Assuming you have the keypoints index for shoulders and hips
        left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx, head_idx = (
            5,
            6,
            11,
            12,
            0,
        )  # Adjust head_idx as needed

        shoulder_width_px = np.linalg.norm(
            np.array(keypoints[left_shoulder_idx][:2])
            - np.array(keypoints[right_shoulder_idx][:2])
        )
        hip_width_px = np.linalg.norm(
            np.array(keypoints[left_hip_idx][:2])
            - np.array(keypoints[right_hip_idx][:2])
        )

        if (
            keypoints[left_shoulder_idx][2] > 0.5
            and keypoints[right_shoulder_idx][2] > 0.5
            and keypoints[left_hip_idx][2] > 0.5
            and keypoints[right_hip_idx][2] > 0.5
        ):
            distance = self.estimate_distance(keypoints)

            # Find a point above the head to draw the text
            head_point = (int(keypoints[head_idx][0]), int(keypoints[head_idx][1]))
            text_position = (
                head_point[0],
                head_point[1] - 20,
            )  # Adjust vertical position as necessary

            # Draw the estimated distance above the head
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5  # Adjust text size as necessary
            color = (255, 255, 255)  # White color
            thickness = 1
            cv2.putText(
                img, f"{distance:.2f} cm", text_position, font, scale, color, thickness
            )

        # Draw connections with colors
        for start_idx, end_idx, color_idx in LIMB_CONNECTIONS:
            if (
                keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5
            ):  # Confidence threshold
                start_point = (
                    int(keypoints[start_idx][0]),
                    int(keypoints[start_idx][1]),
                )
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                color = LIMB_COLORS[color_idx]
                cv2.line(img, start_point, end_point, color, 1)

        # Draw keypoints
        for keypoint in keypoints:
            x, y, conf = keypoint
            if conf > 0.3:  # Confidence threshold
                cv2.circle(
                    img,
                    (int(x), int(y)),
                    0,
                    (255, 255, 255),
                    thickness=-1,
                    lineType=cv2.FILLED,
                )


def main(args=None):
    rclpy.init(args=args)
    image_converter = ImageConverter()
    rclpy.spin(image_converter)
    image_converter.destroy_node()
    rclpy
