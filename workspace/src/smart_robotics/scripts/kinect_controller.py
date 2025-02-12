#!/usr/bin/env python

from multiprocessing import Queue
from pathlib import Path
import signal
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
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3
from custom_msg.msg import PosesWithScales
from object_detection import HardCodedObjectDetector, ContourObjectDetector

class KinectController:
        
    def __init__(self):
        self.bridge = CvBridge()        

        self.K = None


        # Subscribers for RGB and depth images
        self.rgb_sub = Subscriber("/kinect/color/image_raw", Image)
        self.depth_sub = Subscriber("/kinect/depth/image_raw", Image)
        self.camera_info_sub = rospy.Subscriber("/kinect/color/camera_info", CameraInfo, self.camera_info_callback)
        self.tf_listener = tf.TransformListener()
        
        sleep(1)
        self.tf_listener.waitForTransform('world', 'camera_link', rospy.Time(0), rospy.Duration(4.0))
        (trans, rot) = self.tf_listener.lookupTransform('world', 'camera_link', rospy.Time(0))
        self.tf_listener.clear()
        del self.tf_listener

        R = tft.quaternion_matrix(rot)
        t = np.array(trans)
        link_to_world = np.eye(4, dtype=np.float32)
        link_to_world[:3, :3] = R[:3, :3]
        link_to_world[:3, 3] = t

        # Kin: [x->right, y->down, z->fwd] cam_link:[x->fwd, y-> left, z->up]
        kin_to_link = np.array([
            [0., 0., 1., 0.],
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 1.],
        ])  
        self.c2w = link_to_world @ kin_to_link
        
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

        self.processing_thread = threading.Thread(target=self._process)
        self.is_processing = threading.Event()

        self.object_detector = ContourObjectDetector("./templates/template.png")
        #self.object_detector = HardCodedObjectDetector()
        self.publisher = rospy.Publisher('/kinect_controller/detected_poses', PosesWithScales, queue_size=1)


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

    
    def start_processing(self, *args, **kwrags) -> None:
        self.is_processing.set()
        self.processing_thread.start()
                
                
    def stop_processing(self, *args, **kwargs) -> None:
        self.is_processing.clear()
        self.processing_thread.join()
        exit()

    
    def compute_poses_scales(self, bboxes:np.ndarray, depth_image: np.ndarray):
        B = bboxes.shape[0]
        x_center, y_center = bboxes[:, 0], bboxes[:, 1]
        p_screen = np.ones((B, 3), dtype=np.float32)
        p_screen[:, :2] = bboxes[:, :2].astype(np.float32)
        
        depth_center = depth_image[y_center, x_center][..., None]
        depth_table = depth_image[281, 310].repeat(B)[..., None] # TODO: Automize this
        mid_depth = (depth_table + depth_center) / 2

        p_cam = p_screen @ np.linalg.inv(self.K).T
        p_cam *= mid_depth

        p_cam_hom = np.concatenate([p_cam, np.ones((p_cam.shape[0], 1), dtype=np.float32)], axis=-1)
        p_world = p_cam_hom @ self.c2w.T

        orientations = np.zeros((B, 4), dtype=np.float32)
        orientations[:, 0] = 1.

        scales = np.ones((B, 4), dtype=np.float32) * 0.5
        return p_world[:, :3], orientations, scales


    def _process(self):
        while not rospy.is_shutdown() and self.is_processing.is_set():
            
            rgb, depth = self.get_rgbd()
            
            if self.K is None or rgb is None or depth is None:
                sleep(0.1)
                continue
            
            print(f'Processing timestep {self.timestep}')

            bboxes = self.object_detector.detect_box(rgb)
            positions, orientations, scales = self.compute_poses_scales(bboxes, depth)

            assert positions.shape[0] == orientations.shape[0] == scales.shape[0], "Shapes not consisntents."
            N_dets = positions.shape[0]
            if N_dets == 0:
                sleep(0.1)
                continue

            try:
                # Create a list of poses
                msg_poses = []
                msg_scales = []
                for i in range(N_dets):
                    px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
                    qw, qx, qy, qz = orientations[i, 0], orientations[i, 1], orientations[i, 2], orientations[i, 3]
                    sx, sy, sz = scales[i, 0], scales[i, 1], scales[i, 2]

                    pose = Pose(position=Point(px, py, pz), orientation=Quaternion(qx, qy, qz, qw))
                    scale = Vector3(sx, sy, sz)
                    msg_poses += [pose]
                    msg_scales += [scale]

                pose_array = PosesWithScales()
                pose_array.header.stamp = rospy.Time.now()
                pose_array.header.frame_id = 'world'
                pose_array.poses = msg_poses
                pose_array.scales = msg_scales
                self.publisher.publish(pose_array)
                
            except Exception as e:
                print(e)


if __name__ == '__main__':
    rospy.init_node("kinect_controller", log_level=rospy.WARN)

    kinect_controller = KinectController()
    kinect_controller.start_processing()

    signal.signal(signal.SIGINT, kinect_controller.stop_processing)

    while not rospy.is_shutdown():
        rospy.spin()