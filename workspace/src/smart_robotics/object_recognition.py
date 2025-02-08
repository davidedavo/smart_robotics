#!/usr/bin/env python

from pathlib import Path
from time import sleep
import rospy
import cv2
import tf
import tf.transformations as tft
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from object_detection import HardCodedObjectDetector  # Classe per il rilevamento della box (definita separatamente)
from panda_robot import PandaArm


class ObjectRecognition:
    def __init__(self):
        # Inizializza il bridge per la conversione delle immagini
        self.bridge = CvBridge()

        self.panda = PandaArm()
        self.gripper = self.panda.get_gripper()
        self.panda.untuck()
        self.gripper.open()

        # Subscriber per le immagini RGB e di profondità
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
        # Sincronizzazione dei topic
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        # Inizializza l'oggetto di rilevamento
        self.object_detector = HardCodedObjectDetector()        
        self.K = None

        self.kinect_images_dir = Path('./kinect/images')
        self.kinect_images_dir.mkdir(exist_ok=True, parents=True)
        self.kinect_depth_dir = Path('./kinect/depths')
        self.kinect_depth_dir.mkdir(exist_ok=True, parents=True)

        self.timestep = 0

    def camera_info_callback(self, data):
        self.K = np.array(data.K).reshape(3, 3)  # Intrinsic camera matrix
        D = np.array(data.D)  # Distortion coefficients

        assert (D == 0).all(), "Distorsion is not handled"


    def callback(self, rgb_data, depth_data):
        try:
            # Converti le immagini ROS in formato OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, rgb_data.encoding)
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding=depth_data.encoding)

            img_path = self.kinect_images_dir / f'{self.timestep:06d}.png'
            depth_path = self.kinect_depth_dir / f'{self.timestep:06d}.png'
            cv2.imwrite(img_path.as_posix(), rgb_image[..., ::-1])
            depth_image_u16 = (depth_image * 1000).astype(np.uint16)
            cv2.imwrite(depth_path.as_posix(), depth_image_u16)

            if self.c2w is None or self.K is None:
                return

            # Rilevamento della box nelle immagini RGB
            boxes = self.object_detector.detect_box(rgb_image)
            if boxes is not None:
                for box in boxes:
                    # Calcola la posizione 3D della box
                    target_position = self.get_3d_coordinates(box, depth_image)

                    # Comando al robot per muoversi verso la box
                    self.move_robot(target_position)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def get_3d_coordinates(self, rgb_box, depth_image):
        x_center, y_center = rgb_box[0], rgb_box[1]
        p_screen = np.ones((1, 3), dtype=np.float32)
        p_screen[:, :2] = rgb_box[:2].astype(np.float32)
        
        depth_center = depth_image[y_center, x_center]  # Ottieni la profondità al centro della box
        depth_table = depth_image[281, 310] # TODO: Automize this
        mid_depth = (depth_table + depth_center) / 2

        p_cam = p_screen @ np.linalg.inv(self.K).T 

        p_cam *= mid_depth

        p_cam_hom = np.concatenate([p_cam, np.ones((p_cam.shape[0], 1), dtype=np.float32)], axis=-1)

        p_world = p_cam_hom @ self.c2w.T

        return p_world[0, :3]

    def move_robot(self, target_pos, target_quat = np.array([1., 0., 0., 0.])):
        target_pos[-1] += 0.1

        intermediate_pose = target_pos.copy()
        intermediate_pose[-1] += 0.2

        ret, int_joints = self.panda.inverse_kinematics(intermediate_pose, target_quat)
        if ret or True:
            self.panda.move_to_joint_position(int_joints)

        rospy.sleep(1)
        
        ret, tgt_joints = self.panda.inverse_kinematics(target_pos, target_quat)
        if ret or True:
            self.panda.move_to_joint_position(tgt_joints)
        
        rospy.sleep(1)

        object_width = 0.05
        grasp_force = 20.0   
        grasp_speed = 0.05

        success = self.gripper.grasp(width=object_width - 0.01, force=grasp_force, speed=grasp_speed)
        if success:
            print("Oggetto afferrato con successo!")
        else:
            print("Impossibile afferrare l'oggetto.")

        rospy.sleep(2)
        
        

if __name__ == '__main__':
    rospy.init_node('object_recognition', anonymous=True)
    obj_rec = ObjectRecognition()
    rospy.spin()
