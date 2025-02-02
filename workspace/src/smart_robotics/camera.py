#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ImageConverter:
    def __init__(self):
        self.bridge = CvBridge()
        
        # Subscribers for RGB and depth images
        self.rgb_sub = Subscriber("/kinect/color/image_raw", Image)
        self.depth_sub = Subscriber("/kinect/depth/image_raw", Image)
        
        # Synchronize the topics
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

    def callback(self, rgb_data, depth_data):
        try:
            # Convert the ROS Image messages to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, rgb_data.encoding)
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding=depth_data.encoding)
            rgb_image = rgb_image.astype(np.float32) / 255.
        except CvBridgeError as e:
            rospy.logerr(e)

        # Display the images
        # cv2.imshow("RGB Image", rgb_image)
        # cv2.imshow("Depth Image", depth_image)
        # cv2.waitKey(3)

def main():
    rospy.init_node('image_converter', anonymous=True)
    ic = ImageConverter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()