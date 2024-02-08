import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result


class ImageConverter(Node):
    def __init__(self):
        super().__init__("image_converter")
        self.subscription = self.create_subscription(
            Image, "/theta_z1/rgb", self.listener_callback, 10
        )
        self.bridge = CvBridge()

    def listener_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Initialize mmpose model
        config_file = "src/object_detection/object_detection/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
        checkpoint_file = "csrc/object_detection/object_detection/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
        model = init_pose_model(config_file, checkpoint_file, device="cpu")

        # Perform pose detection
        pose_results, _ = inference_top_down_pose_model(model, cv_image)

        # Visualize pose detection results
        vis_img = vis_pose_result(
            model, cv_image, pose_results, dataset="TopDownCocoDataset"
        )

        # Convert image back to ROS message
        converted_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")

        # Publish the image with pose detection
        self.image_publisher.publish(converted_msg)


def main(args=None):
    rclpy.init(args=args)
    image_converter = ImageConverter()
    rclpy.spin(image_converter)
    image_converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
