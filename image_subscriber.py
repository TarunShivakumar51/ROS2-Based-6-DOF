#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesis
from vision_msgs.msg import ObjectHypothesisWithPose
from .pose_estimation import estimate_pose
from ultralytics import YOLO
import numpy as np
import quaternion as quat
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster

class ImageSubscriberNode(Node):
    
    def __init__(self):
        super().__init__("pose_subscriber")

        self.tvec = []
        self.rvec = []
        self.q_components = np.array([])

        self.bridge = CvBridge()

        self.reference = "/home/tarun2006/ros2_ws/src/perception_projects/perception_projects/Scene2.png"
        
        self.camera_sub_ = self.create_subscription(Image, "/camera/image_raw", self.receive_image, 1)

        self.vision_msgs_pub_ = self.create_publisher(Detection3DArray, "/detections_3d", 1)

        self.marker_pub_ = self.create_publisher(Marker, "/pose_maker", 1)

        # self.pose_pub_ = self.create_publisher(Pose, "/pose", 1)

        self.pose_stamped_pub_ = self.create_publisher(PoseStamped, "/pose_stamped", 1)

        model_path = os.path.join(get_package_share_directory('perception_projects'), 'rs25_8_15_2.pt')
        self.model = YOLO(model_path)

        # self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)

    def receive_image(self, msg : Image):
        self.get_logger().info("Image Received")
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8') 

        results = self.model.predict(cv_image, save=True, imgsz=320, conf=0.25)
        result = results[0]

        err, pose_est = estimate_pose(self.reference, cv_image)
        if err is not None or pose_est is None:
            self.get_logger().warn(f"Pose estimation failed: {err}")
            return

        array = Detection3DArray()
        array.header = msg.header

        det = Detection3D()
        det.header = msg.header
        
        best_box = None
        best_conf = -1.0

        for box in result.boxes:
            conf = box.conf[0].item()  
            if conf > best_conf:
                best_conf = conf
                best_box = box

        threshold = 0.9
        if best_conf < threshold:
            return        

        obj_hyp = ObjectHypothesis()
        obj_hyp.class_id = result.names[best_box.cls[0].item()]
        obj_hyp.score = best_conf

        self.rvec = pose_est['rvec']
        self.tvec = pose_est['tvec'].flatten()

        q = quat.from_rotation_vector(self.rvec.flatten())
        self.q_components = quat.as_float_array(q)

        self.pose()

        obj_hyp_pose = ObjectHypothesisWithPose()

        obj_hyp_pose.pose.pose.position.x = float(self.tvec[0])
        obj_hyp_pose.pose.pose.position.y = float(self.tvec[1])
        obj_hyp_pose.pose.pose.position.z = float(self.tvec[2])

        obj_hyp_pose.pose.pose.orientation.w = self.q_components[0]
        obj_hyp_pose.pose.pose.orientation.x = self.q_components[1]
        obj_hyp_pose.pose.pose.orientation.y = self.q_components[2]
        obj_hyp_pose.pose.pose.orientation.z = self.q_components[3]

        obj_hyp_pose.hypothesis = obj_hyp

        det.results.append(obj_hyp_pose)

        array.detections.append(det)

        self.vision_msgs_pub_.publish(array)

        marker = Marker()

        marker.header = msg.header
        marker.header.frame_id = "pose"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detections"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = obj_hyp_pose.pose.pose
        marker.scale.x = 20.0
        marker.scale.y = 3.0
        marker.scale.z = 3.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub_.publish(marker)

        pose_stamped = PoseStamped()

        pose_stamped.header.frame_id = "poster"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        
        pose = Pose()

        point = Point()
        point.x = float(self.tvec[0])
        point.y = float(self.tvec[1])
        point.z = float(self.tvec[2])

        quaternion = Quaternion()
        quaternion.w = self.q_components[0]
        quaternion.x = self.q_components[1]
        quaternion.y = self.q_components[2]
        quaternion.z = self.q_components[3]

        pose.position = point
        pose.orientation = quaternion

        pose_stamped.pose = pose

        self.pose_stamped_pub_.publish(pose_stamped)

    def pose(self):
        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "poster"
        t.child_frame_id = "camera"

        t.transform.translation.x = float(self.tvec[0])
        t.transform.translation.y = float(self.tvec[1])
        t.transform.translation.z = float(self.tvec[2])

        # t.transform.translation.x = 0.0
        # t.transform.translation.y = 0.0
        # t.transform.translation.z = 0.0

        t.transform.rotation.w = self.q_components[0]
        t.transform.rotation.x = self.q_components[1]
        t.transform.rotation.y = self.q_components[2]
        t.transform.rotation.z = self.q_components[3]

        self.tf_broadcaster.sendTransform(t)    

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()