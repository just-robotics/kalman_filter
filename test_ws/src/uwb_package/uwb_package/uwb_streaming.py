import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

# import numpy as np

# from uwb_package.submodules.readSensorData import readSensorData


class UWBCoordsStreamer(Node):
    def __init__(self):
        super().__init__('uwb_streamer')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 9000)
        self.declare_parameter('reconnect_interval', 1)
        self.declare_parameter('max_no_data_time', 5)

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.reconnect_interval = self.get_parameter('reconnect_interval').get_parameter_value().integer_value
        self.max_no_data_time = self.get_parameter('max_no_data_time').get_parameter_value().integer_value

        self.get_logger().info(f"port: {self.port}")
        self.get_logger().info(f"baudrate: {self.baudrate}")
        self.get_logger().info(f"reconnect_interval: {self.reconnect_interval}")
        self.get_logger().info(f"max_no_data_time: {self.max_no_data_time}")

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            'uwb_coordinates',
            10
        )
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()

        # TODO: Write code for readSensorData function

        x = 0
        y = 0
        msg.data = [x, y]
        self.publisher_.publish(msg)
        self.get_logger().info(f'uwb_coordinates: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    uwb_publisher = UWBCoordsStreamer()

    rclpy.spin(uwb_publisher)

    uwb_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()