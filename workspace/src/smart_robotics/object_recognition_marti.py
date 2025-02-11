#!/usr/bin/env python

from pathlib import Path
from time import sleep
import rospy
import cv2
import tf
import tf.transformations as tft
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from object_detection import HardCodedObjectDetector  # Classe per il rilevamento della box (definita separatamente)
from panda_robot import PandaArm
from object_detector_marti import ObjectDetector, ObjectDetector_YOLO
from scripts.kinect_controller import KinectController



class ObjectRecognition:
    def __init__(self):
        # Inizializza il bridge per la conversione delle immagini
        self.bridge = CvBridge()

        self.panda = PandaArm()
        self.gripper = self.panda.get_gripper()
        self.panda.untuck()
        self.gripper.open()

        self.kinect_controller = KinectController()  # Aggiungi KinectController
        
        self.object_detector = ObjectDetector()   # Inizializza il rilevatore di oggetti

        self.camera_info_sub = rospy.Subscriber("/kinect/color/camera_info", CameraInfo, self.camera_info_callback)
        # Inizializza il publisher per la posizione dell'oggetto
        self.object_pose_pub = rospy.Publisher('/detected_object_pose', Pose, queue_size=10)
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
        self.kinect_controller.set_rgbd(rgb=None, depth=None)  # Inizializza le immagini a None

        self.kinect_images_dir = Path('./kinect/images')
        self.kinect_images_dir.mkdir(exist_ok=True, parents=True)
        self.kinect_depth_dir = Path('./kinect/depths')
        self.kinect_depth_dir.mkdir(exist_ok=True, parents=True)

        self.timestep = 0

        rospy.Timer(rospy.Duration(0.1), self.timer_callback)  # Chiama periodicamente il metodo per aggiornare


    def camera_info_callback(self, data):
        self.K = np.array(data.K).reshape(3, 3)  # Intrinsic camera matrix
        D = np.array(data.D)  # Distortion coefficients

        assert (D == 0).all(), "Distorsion is not handled"


    def timer_callback(self, event):
        # Ottieni le immagini RGB e di profondità dal KinectController
        rgb_image, depth_image = self.kinect_controller.get_rgbd()

        if rgb_image is not None and depth_image is not None:
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

                    self.publish_object_pose(target_position)

                    # Comando al robot per muoversi verso la box
                    #self.move_robot(target_position)


    def publish_object_pose(self, target_position):
        # Crea il messaggio di tipo Pose per ROS
        pose = Pose()
        pose.position.x = target_position[0]
        pose.position.y = target_position[1]
        pose.position.z = target_position[2]
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0  # Orientamento neutro

        # Pubblica il messaggio
        self.object_pose_pub.publish(pose)


    def get_3d_coordinates(self, rgb_boxes, depth_image):
        # Lista per memorizzare le coordinate 3D di tutti i box
        coordinates_3d = []

        for rgb_box in rgb_boxes:
            # Prendi il centro del bounding box
            x_center, y_center = rgb_box[0], rgb_box[1]
            
            # Crea il punto nella schermata (punto 2D)
            p_screen = np.ones((1, 3), dtype=np.float32)
            p_screen[:, :2] = rgb_box[:2].astype(np.float32)
            
            # Ottieni la profondità al centro del box
            depth_center = depth_image[y_center, x_center]  # Profondità al centro
            depth_table = depth_image[281, 310]  # Valore di profondità di riferimento, potrebbe essere automatizzato
            mid_depth = (depth_table + depth_center) / 2  # Media delle profondità
            
            # Proietta il punto 2D (schermo) in 3D nello spazio della fotocamera
            p_cam = p_screen @ np.linalg.inv(self.K).T 
            p_cam *= mid_depth
            
            # Trasformazione per ottenere il punto nello spazio del mondo
            p_cam_hom = np.concatenate([p_cam, np.ones((p_cam.shape[0], 1), dtype=np.float32)], axis=-1)
            p_world = p_cam_hom @ self.c2w.T
            
            # Aggiungi il risultato alla lista
            coordinates_3d.append(p_world[0, :3])
        
        return coordinates_3d
        
        

if __name__ == '__main__':
    rospy.init_node('object_recognition', anonymous=True)
    obj_rec = ObjectRecognition()
    rospy.spin()
