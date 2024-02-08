import rclpy
import time
from sensor_msgs.msg import Image

data = list()


def callback(msg):
    data.append(msg.data)
    if len(data) == 100:
        print(data)
        return


# Data in numpy oder anderes format abspeichern mit frame id damit wir anderes damit machen koennen
def main():
    current_time = time.time()
    rclpy.init()

    node = rclpy.create_node("record")
    topic_name = "/theta_z1/rgb"

    msg_type = Image

    subscription = node.create_subscription(msg_type, topic_name, callback, 10)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
