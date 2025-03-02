#!/usr/bin/env python

from multiprocessing import Queue
from pathlib import Path
import signal
import threading
from time import sleep
from typing import List, Tuple
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf
import tf.transformations as tft
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3
from custom_msg.msg import Detection3D
from scipy.spatial.transform import Rotation as R
from object_detection import HardCodedObjectDetector, ContourObjectDetector


def backproject_points(points_2d:np.ndarray, depths:np.ndarray, K:np.ndarray, c2w:np.ndarray):
    B = points_2d.shape[0]
    points_2d = points_2d.astype(np.float32)
    points_2d_hom = np.concatenate([points_2d, np.ones((B, 1)).astype(np.float32)], axis=-1)
    p_cam = points_2d_hom @ np.linalg.inv(K).T
    p_cam *= depths

    p_cam_hom = np.concatenate([p_cam, np.ones((p_cam.shape[0], 1), dtype=np.float32)], axis=-1)
    p_world = p_cam_hom @ c2w.T
    return p_world[:, :3]

def compute_rotation(rects):
    rots = []
    for rect in rects:
        angle = rect[-1]
        rotation_z = np.deg2rad(angle)  # Convert to radians
        Rz = np.array([
            [np.cos(rotation_z), -np.sin(rotation_z), 0],
            [np.sin(rotation_z),  np.cos(rotation_z), 0],
            [0, 0, 1]
        ])
        rots.append(Rz)
    rots = np.stack(rots, axis=0)
    return rots


def permute_rect_points(bbox_points):
    masks = []
    p_center = bbox_points.mean(axis=0, keepdims=True)
    delta = bbox_points - p_center
    masks += [(delta[..., 0] < 0) & (delta[..., 1] < 0)]
    masks += [(delta[..., 0] >= 0) & (delta[..., 1] < 0)]
    masks += [(delta[..., 0] >= 0) & (delta[..., 1] >= 0)]
    masks += [(delta[..., 0] < 0) & (delta[..., 1] >= 0)]
    
    masks = np.array(masks)
    perms = np.nonzero(np.array(masks))[1]
    return bbox_points[perms]


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

        self.kinect_images_dir = Path('/workspace/kinect/images')
        self.kinect_images_dir.mkdir(exist_ok=True, parents=True)
        self.kinect_depth_dir = Path('/workspace/kinect/depths')
        self.kinect_depth_dir.mkdir(exist_ok=True, parents=True)

        # Do not access following attributes directly. Use the setter and getter.
        self._next_timestep = 0
        self.image_queue = Queue(maxsize=1)

        self._images_lock = threading.Lock()

        self.rate = rospy.Rate(30)
        

        self.object_detector = ContourObjectDetector("./templates/template.png")
        #self.object_detector = HardCodedObjectDetector()
        self.det_publisher = rospy.Publisher('/kinect_controller/detected_poses', Detection3D, queue_size=1)
        self.image_publisher = rospy.Publisher('/kinect_controller/detected_images', Image, queue_size=1)

        # Synchronize the topics
        self.images_ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.images_ts.registerCallback(self.images_callback)
        


    def set_image_data(self, rgb, depth):
        with self._images_lock:
            try:
                if self.image_queue.full():
                    self.image_queue.get(block=True)
                self.image_queue.put({'rgb':rgb, 'depth': depth, 'timestep': self._next_timestep})
            except Exception as e:
                print(e)
                pass
            self._next_timestep += 1

    def get_image_data(self):
        data = None
        with self._images_lock:
            if not self.image_queue.empty():
                data = self.image_queue.get()                
        return data
        
    
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
        if False:
            img_path = self.kinect_images_dir / f'{self.timestep:06d}.png'
            depth_path = self.kinect_depth_dir / f'{self.timestep:06d}.png'
            cv2.imwrite(img_path.as_posix(), (rgb_image[..., ::-1]*255).astype(np.uint8))
            depth_image_u16 = (depth_image * 1000).astype(np.uint16)
            cv2.imwrite(depth_path.as_posix(), depth_image_u16)

        self.set_image_data(rgb_image, depth_image)

    
    # def start_processing(self, *args, **kwrags) -> None:
    #     self.is_processing.set()
    #     # self.processing_thread.start()
                
                
    # def stop_processing(self, *args, **kwargs) -> None:
    #     self.is_processing.clear()
    #     # self.processing_thread.join()
    #     exit()

    def is_in_area(self, rects, target_box):
        if len(rects) == 0:
            return rects        
        x_center = np.array([rect[0][0] for rect in rects], dtype=np.float32)
        y_center = np.array([rect[0][1] for rect in rects], dtype=np.float32)
        # x_center, y_center = rects[:, 0], rects[:, 1]

        xi_target = target_box[0, 0]  - target_box[0, 2] / 2
        yi_target = target_box[0, 1]  - target_box[0, 3] / 2
        xf_target = target_box[0, 0]  + target_box[0, 2] / 2
        yf_target = target_box[0, 1]  + target_box[0, 3] / 2

        mask = (x_center > xi_target) & (x_center < xf_target) & (y_center > yi_target) & (y_center < yf_target)
        return mask
        
    def get_target_box(self, w, h, area_ratio=0.05):
        x_c = w / 2
        y_c = h / 2
        
        w = w*area_ratio
        h = h*area_ratio
        return np.array([[x_c, y_c, w, h]])
    
    # def get_non_detection_pixels(self, image, bboxes, margin=20):
    #     H, W = image.shape[:2]
    #     mask = np.ones((H, W), dtype=bool)
        
    #     for bbox in bboxes:
    #         xc, yc, w, h = bbox
    #         x1 = max(int(xc - w / 2 - margin), 0)
    #         y1 = max(int(yc - h / 2 - margin), 0)
    #         x2 = min(int(xc + w / 2 + margin), W)
    #         y2 = min(int(yc + h / 2 + margin), H)
    #         mask[y1:y2, x1:x2] = False
        
    #     non_detection_pixels = image[mask]
    #     return non_detection_pixels

    def get_non_detection_pixels(self, image, rects, margin=20):
        H, W = image.shape[:2]
        mask = np.ones((H, W), dtype=bool)
        
        for rect in rects:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Create a mask for the rotated rectangle
            rect_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(rect_mask, [box], 255)
            
            # Apply margin by dilating the mask
            kernel = np.ones((margin, margin), np.uint8)
            rect_mask = cv2.dilate(rect_mask.astype(np.uint8), kernel, iterations=1)
            rect_mask = rect_mask > 127

            # Combine the masks
            mask &= ~rect_mask
        
        non_detection_pixels = image[mask]
        return non_detection_pixels
    
    def compute_poses_scales(self, rects:List[Tuple], depth_image: np.ndarray):
        B = len(rects)
        if B == 0:
            return None, None, None
        x_center = np.array([rect[0][0] for rect in rects], dtype=np.int64)
        y_center = np.array([rect[0][1] for rect in rects], dtype=np.int64)
        depth_object = depth_image[y_center, x_center][..., None]

        non_detection_depths = self.get_non_detection_pixels(depth_image, rects)
        depth_table = np.median(non_detection_depths).repeat(B)[..., None]
        
        mid_depth = (depth_table + depth_object) / 2
        
        # p_center = bboxes[:, :2].copy().astype(np.float32)
        # p_left_top = p_center - bboxes[:, 2:4] / 2
        # p_right_bottom = p_center + bboxes[:, 2:4] / 2

        p_center = np.array([[rect[0][0], rect[0][1]] for rect in rects], dtype=np.float32)
        bboxes_points = np.array([cv2.boxPoints(rect) for rect in rects], dtype=np.float32)

        for i in range(bboxes_points.shape[0]):
            perm_points = permute_rect_points(bboxes_points[i])
            bboxes_points[i] = perm_points

        # p_left_top = np.array([cv2.boxPoints(rect)[2] for rect in rects], dtype=np.float32)
        # p_right_bottom = np.array([cv2.boxPoints(rect)[0] for rect in rects], dtype=np.float32)

        points_shape  = bboxes_points.shape
        bboxes_points = bboxes_points.reshape(-1, 2)
        
        # Points in world coordinate
        p_center_w = backproject_points(p_center, depth_object, self.K, self.c2w)

        mid_depth = mid_depth[:, None].repeat(4, axis=1).reshape(-1, 1)
        p_bboxes_w = backproject_points(bboxes_points, mid_depth, self.K, self.c2w)

        p_bboxes_w = p_bboxes_w.reshape(*points_shape[:2], 3)

        # Point are expressed as [TL, TR, BR, BL]
        scales = np.zeros((B, 3), dtype=np.float32)
        a = p_bboxes_w[:, 1] - p_bboxes_w[:, 0]
        b = p_bboxes_w[:, 3] - p_bboxes_w[:, 0]
        a_norm = np.linalg.norm(a, axis=-1)[..., None]
        b_norm = np.linalg.norm(b, axis=-1)[..., None]
        a_dir = a / (a_norm + 1e-10)
        b_dir = b / (b_norm + 1e-10)

        R_objects = np.eye(3)[None].repeat(B, axis=0)
        mask = (a_norm > b_norm)#[..., None]
        R_objects[:, 0, :] = np.where(mask, a_dir, b_dir)
        R_objects[:, 1, :] = np.cross(R_objects[:, 2], R_objects[:, 0])
        scales[:, [0]] = np.where(mask, a_norm, b_norm)
        scales[:, [1]] = np.where(mask, b_norm, a_norm)
 
        scales[:, [2]] = depth_table - depth_object

        # R = np.eye(4)
        quats = R.from_matrix(R_objects).as_quat()

        # Convert the batch of rotation matrices to quaternions
        # quats = rotations.as_quat()
        quats_wxyz = quats[:, [3, 0, 1, 2]]
        
        # identity_quat = np.zeros((B, 4), dtype=np.float32)
        # identity_quat[:, 0] = 1.

        # angles = np.array([rect[2] for rect in rects])
        # mask = np.abs(angles) == 90
        # if mask.sum() > 0:
        #     quats_wxyz[mask] = identity_quat[mask]

        return p_center_w, quats_wxyz, scales

    def process(self):
        while not rospy.is_shutdown():
            data = self.get_image_data()
            if data is None:
                self.rate.sleep()
                continue
            
            rgb, depth, timestep = data['rgb'], data['depth'], data['timestep']
            
            if self.K is None or rgb is None or depth is None:
                self.rate.sleep()
                continue
            
            rospy.loginfo(f'Processing timestep {timestep}')

            # Otteniamo le bbox e la lista dei flag (True se quadrato, False se non quadrato)
            rects, classes_ids = self.object_detector.detect_box(rgb)
            target_box = self.get_target_box(rgb.shape[1], rgb.shape[0])

            # Disegna tutte le bbox sull'immagine (verde per quadrato, rosso per non quadrato)
            overlay = rgb.copy()
            overlay = (overlay * 255).astype(np.uint8)
            overlay = draw_rects(overlay, rects, classes_ids)
            overlay = draw_bboxes(overlay, target_box, [3])
            
            ros_image = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            self.image_publisher.publish(ros_image)

            positions, orientations, scales = self.compute_poses_scales(rects, depth)
            
            if positions is None:
                self.rate.sleep()
                continue

            assert positions.shape[0] == orientations.shape[0] == scales.shape[0], "Shapes non consistenti."
            N_dets = positions.shape[0]
            if N_dets == 0:
                self.rate.sleep()
                continue

            filtered_rects = []
            if len(rects) > 0:
                mask = self.is_in_area(rects, target_box)
                filtered_rects = [rect for i,rect in enumerate(rects) if mask[i]]
                classes_ids = [cid for i,cid in enumerate(classes_ids) if mask[i]]
                positions = positions[mask]
                scales = scales[mask]
                orientations = orientations[mask]
                # filtered_bboxes = rects[mask]
                # classes_ids = classes_ids[mask]
            if len(filtered_rects) == 0:
                self.rate.sleep()
                continue
            
            filtered_rect = filtered_rects[0]
            class_id = int(classes_ids[0])

            try:
                # msg_poses = []
                # msg_scales = []
                # we assume only one detection at time
                px, py, pz = positions[0, 0], positions[0, 1], positions[0, 2]
                qw, qx, qy, qz = orientations[0, 0], orientations[0, 1], orientations[0, 2], orientations[0, 3]
                sx, sy, sz = scales[0, 0], scales[0, 1], scales[0, 2]

                pose = Pose(position=Point(px, py, pz), orientation=Quaternion(qx, qy, qz, qw))
                scale = Vector3(sx, sy, sz)
                # msg_poses.append(pose)
                # msg_scales.append(scale)

                pose_array = Detection3D()
                pose_array.header.stamp = rospy.Time.now()
                pose_array.header.frame_id = 'world'
                pose_array.pose = pose
                pose_array.scale = scale
                pose_array.class_id = class_id
                self.det_publisher.publish(pose_array)
                    
            except Exception as e:
                print(e)
            
            self.rate.sleep()


COLORS = [
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0)
]

def draw_bboxes(image, bboxes, classes_ids):
    for i, (bbox, class_id) in enumerate(zip(bboxes, classes_ids)):
        xc, yc, w, h = bbox
        x = int(xc - w // 2)
        y = int(yc - h // 2)
        w = int(w)
        h = int(h)
        color = COLORS[int(class_id)]
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # image = cv2.putText(image, str(bbox), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def draw_rects(image, rects, classes_ids):
    for i, (rect, class_id) in enumerate(zip(rects, classes_ids)):
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        color = COLORS[int(class_id)]
        image = cv2.drawContours(image, [box], 0, color, 2)
    return image


if __name__ == '__main__':
    rospy.init_node("kinect_controller", log_level=rospy.WARN)

    kinect_controller = KinectController()
    # kinect_controller.start_processing()

    # signal.signal(signal.SIGINT, kinect_controller.stop_processing)

    kinect_controller.process()
    # while not rospy.is_shutdown():
    #     rospy.spin()