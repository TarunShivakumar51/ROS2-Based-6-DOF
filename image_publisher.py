import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class PublishImage(Node):
    def __init__(self):
        super().__init__("pose_publisher")
        
        self.camera_pub_ = self.create_publisher(Image, "/camera/image_raw", 10)
        
        video_path = "/home/tarun2006/ros2_ws/src/perception_projects/perception_projects/IMG_0376.MOV" 
        self.vid = cv2.VideoCapture(video_path)

        self.timer = self.create_timer(0.1, self.send_image)
        self.bridge = CvBridge()

        self.get_logger().info("Sending image")

        # self.frame_count = 0

    def send_image(self):

        success, img = self.vid.read() 
        # self.frame_count += 1 

        if success:
            cv_image = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            self.camera_pub_.publish(cv_image)
        else:
            self.get_logger().info("Didn't work")

        # if self.frame_count == int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)):
        #     self.frame_count = 0
        #     self.vid.set(cv2.CV_CAP_PROP_POS_FRAMES, 0)        

def main(args=None):
    rclpy.init(args=args)
    node = PublishImage()
    rclpy.spin(node)
    rclpy.shutdown()
