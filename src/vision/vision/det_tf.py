# Subscribe Detection result and broadcast tf
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformStamped
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point
import time

class DetTF(Node):
    def __init__(self):
        super().__init__('det_tf')
        self.declare_parameter('class_names', ['red_box', 'blue_box'])
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.class_names = list(self.get_parameter('class_names').value)
        self.camera_frame = self.get_parameter('camera_frame').value

        self.subscription = self.create_subscription(
            Detection2DArray,
            'detection',
            self.det_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.tf_broadcaster = TransformBroadcaster(self)

    def det_callback(self, msg: Detection2DArray):
        objects_by_class = {class_name: [] for class_name in self.class_names}

        for det in msg.detections:
            if det.id in objects_by_class and det.results:
                objects_by_class[det.id].append(det.results[0].pose.pose.position)

        frame_id = msg.header.frame_id or self.camera_frame

        for class_name, objects in objects_by_class.items():
            objects.sort(key=lambda p: p.x)

            for idx, pos in enumerate(objects, start=1):
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = frame_id
                t.child_frame_id = f'{class_name}{idx}'

                t.transform.translation.x = pos.x
                t.transform.translation.y = pos.y
                t.transform.translation.z = pos.z
                t.transform.rotation.w = 1.0
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0

                self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)

    det_tf = DetTF()

    try:
        rclpy.spin(det_tf)
    except KeyboardInterrupt:
        pass

    det_tf.destroy_node()
    rclpy.shutdown()
