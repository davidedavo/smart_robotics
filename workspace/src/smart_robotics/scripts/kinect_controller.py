#!/usr/bin/env python

from pathlib import Path
import threading
from time import sleep
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf
import tf.transformations as tft
from sensor_msgs.msg import Image, CameraInfo

class KinectController:
        
    def __init__(self):
        self.bridge = CvBridge()        
        # Subscribers for RGB and depth images
        self.rgb_sub = Subscriber("/kinect/color/image_raw", Image)
        self.depth_sub = Subscriber("/kinect/depth/image_raw", Image)
        
        self.camera_info_sub = rospy.Subscriber("/kinect/color/camera_info", CameraInfo, self.camera_info_callback)
        self.tf_listener = tf.TransformListener()
        
        # sleep(1)
        # self.tf_listener.waitForTransform('world', 'camera_link', rospy.Time(0), rospy.Duration(4.0))
        # (trans, rot) = self.tf_listener.lookupTransform('world', 'camera_link', rospy.Time(0))
        # self.tf_listener.clear()
        # del self.tf_listener
        
        # Synchronize the topics
        self.images_ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.images_ts.registerCallback(self.images_callback)

        self.kinect_images_dir = Path('./kinect/images')
        self.kinect_images_dir.mkdir(exist_ok=True, parents=True)
        self.kinect_depth_dir = Path('./kinect/depths')
        self.kinect_depth_dir.mkdir(exist_ok=True, parents=True)

        # Do not access following attributes directly. Use the setter and getter.
        self._rgb = None 
        self._depth = None 
        self._timestep = 0

        self._images_lock = threading.Lock()


    def set_rgbd(self, rgb, depth):
        with self._images_lock:
            self._rgb = rgb
            self._depth = depth
            self._timestep += 1

    def get_rgbd(self):
        with self._images_lock:
            return self._rgb, self._depth
        
    @property
    def timestep(self):
        with self._images_lock:
            return self._timestep
    
    def camera_info_callback(self, data):
        self.K = np.array(data.K).reshape(3, 3)  # Intrinsic camera matrix
        D = np.array(data.D)  # Distortion coefficients
        assert (D == 0).all(), "Distorsion is not handled"

    
    def images_callback(self, rgb_data, depth_data):
        try:
            # Convert the ROS Image messages to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, rgb_data.encoding)
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding=depth_data.encoding)
            rgb_image = rgb_image.astype(np.float32) / 255.
        except CvBridgeError as e:
            rospy.logerr(e)

        # store images on filesystem
        img_path = self.kinect_images_dir / f'{self.timestep:06d}.png'
        depth_path = self.kinect_depth_dir / f'{self.timestep:06d}.png'
        cv2.imwrite(img_path.as_posix(), (rgb_image[..., ::-1]*255).astype(np.uint8))
        depth_image_u16 = (depth_image * 1000).astype(np.uint16)
        cv2.imwrite(depth_path.as_posix(), depth_image_u16)

        self.set_rgbd(rgb_image, depth_image)


    def __call__(self):
        pass


if __name__ == '__main__':
    rospy.init_node("kinect_controller")

    kinect_controller = KinectController()

    while not rospy.is_shutdown():
        rospy.spin()