#!/usr/bin/env python

import math
from multiprocessing import Queue
from queue import Full
import signal
import threading
from time import sleep

import numpy as np
import rospy
from panda_robot import PandaArm
import tf
import tf.transformations as tft
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
from custom_msg.msg import PosesWithScales
from custom_msg.srv import PickObject, PickObjectResponse


def calculate_rotation(start, target):
    delta_x = target[0] - start[0]
    delta_y = target[1] - start[1]
    
    angle_rad = math.atan2(delta_y, delta_x)
    
    return angle_rad


def get_transform_matrix(t, q):
    R = tft.quaternion_matrix(q)
    t = np.array(t)
    trans_matrix = np.eye(4, dtype=np.float32)
    trans_matrix[:3, :3] = R[:3, :3]
    trans_matrix[:3, 3] = t
    return trans_matrix



class PandaController:
    def __init__(self):
        

        self.data_queue = Queue(maxsize=100)

        self.panda = PandaArm()
        self.gripper = self.panda.get_gripper()
        self.gripper.open()
        self.panda.untuck()
        
        # self.is_processing = threading.Event()
        # self.is_processing.set()
        # self.processing_thread = threading.Thread(target=self._process)
        sleep(1)
        self.pick_and_place_service = rospy.Service('/panda/pick_and_place', PickObject, self.handle_pick_place_service)
        self.status_publisher = rospy.Publisher('/panda/pick_status', String, queue_size=10) # ["idle", "grasping", "releasing"]

        self.tf_listener = tf.TransformListener()
        self.tf_listener.waitForTransform('panda_hand', 'panda_link0', rospy.Time(0), rospy.Duration(4.0))
        self.tf_listener.waitForTransform('panda_link0', 'panda_rightfinger', rospy.Time(0), rospy.Duration(4.0))
        self.tf_listener.waitForTransform('panda_link0', 'panda_leftfinger', rospy.Time(0), rospy.Duration(4.0))
        

        # T_lf_to_0 = get_transform_matrix(lf_t, lf_q)
        # T_rf_to_0 = get_transform_matrix(rf_t, rf_q)
        # T_0_to_hand = get_transform_matrix(*hand_transforms)
        pass
        
        # self.objects_poses_subscriber = rospy.Subscriber("/kinect_controller/detected_poses", PosesWithScales, self.poses_callback)
        # self.processing_thread.start()

    def handle_pick_place_service(self, req):
        pick_pos = np.array([req.pose.position.x, req.pose.position.y, req.pose.position.z])
        orientation = np.array([req.pose.orientation.w, req.pose.orientation.x, req.pose.orientation.y, req.pose.orientation.z])
        scale = np.array([req.scale.x, req.scale.y, req.scale.z])
        release_pos = np.array([req.place_pos.x, req.place_pos.y, req.place_pos.z])
        if not self.data_queue.full():
            self.data_queue.put({'pick_pos': pick_pos, 'orient': orientation, 'scale': scale, 'release_pos': release_pos}, block=False, timeout=0.01)
        else:
            pass # Don't do anything we skip these detections.
        return True
    

    def grasp_task(self, target_pos, target_quat = np.array([1., 0., 0., 0.]), target_scales=np.array([0.05, 0.05, 0.05])):
        self.status_publisher.publish('grasping')
        
        # rf_t, rf_q = self.tf_listener.lookupTransform('panda_link0', 'panda_rightfinger', rospy.Time(0))
        # lf_t, lf_q = self.tf_listener.lookupTransform('panda_link0', 'panda_leftfinger', rospy.Time(0))
        # ph_t, ph_q = self.tf_listener.lookupTransform('panda_link0', 'panda_hand', rospy.Time(0))
        
        # R_w_h = tft.quaternion_matrix(ph_q)[:3, :3]

        target_quat_xyzw = target_quat[[1, 2, 3, 0]]
        target_rot = R.from_quat(target_quat_xyzw).as_matrix()

        T_w_e = np.eye(3)
        T_w_e[[1, 2]] *= -1
        ee_rot = T_w_e @ target_rot
        ee_quat = R.from_matrix(ee_rot).as_quat()

        identity_quat = np.array([1., 0., 0., 0.])
        target_pos[-1] += 0.09
        target_pos[0] += 0.01

        intermediate_pose = target_pos.copy()
        intermediate_pose[-1] += 0.2

        ret, int_joints = self.panda.inverse_kinematics(intermediate_pose, ee_quat)
        if ret or True:
            self.panda.move_to_joint_position(int_joints)
        
        
        ret, tgt_joints = self.panda.inverse_kinematics(target_pos, ee_quat)
        if ret or True:
            self.panda.move_to_joint_position(tgt_joints)
        
        rospy.sleep(0.01)

        object_width = target_scales[1]
        grasp_force = 10.0
        grasp_speed = 0.05

        success = self.gripper.grasp(width=object_width - 0.005, force=grasp_force, speed=grasp_speed)
        if success:
            print("Oggetto afferrato con successo!")
        else:
            print("Impossibile afferrare l'oggetto.")

        rospy.sleep(0.01)

    def release_task(self, target_pos, target_quat = np.array([1., 0., 0., 0.])):
        
        self.panda.untuck()

        self.status_publisher.publish('releasing')
        
        # target_pos[-1] = intermediate_pos[-1]
        # ret, tgt_joints = self.panda.inverse_kinematics(target_pos, ee_rot)
        # if ret or True:
        #     self.panda.move_to_joint_position(tgt_joints)
        current_pose, _ = self.panda.ee_pose()
        angle_z = calculate_rotation(current_pose, target_pos)
        a = self.panda.angles() * 0
        a[0] = angle_z 
        self.panda.move_to_joint_pos_delta(a)

        angles = self.panda.angles()
        ee_pos, ee_rot = self.panda.forward_kinematics(angles)
        target_pos[2] = ee_pos[2]
        ret, tgt_joints = self.panda.inverse_kinematics(target_pos, ee_rot)
        if ret or True:
            self.panda.move_to_joint_position(tgt_joints)

        self.gripper.open()

        self.panda.move_to_joint_position(angles)
        self.panda.untuck()

        rospy.sleep(0.01)

    
    def process(self):
        while not rospy.is_shutdown():
            if self.data_queue.empty():
                sleep(0.01)
                continue
            
            data = self.data_queue.get(block=False)
            
            if data is None:
                sleep(0.01)
                continue

            position = data['pick_pos']
            scale = data['scale']
            orient = data['orient']
            release_position = data['release_pos']
                
            # Do all the task here
            self.grasp_task(position, orient, scale)

            self.release_task(release_position)
            self.status_publisher.publish('idle')

            # # Clean the queue from old predictions
            # if self.data_queue.full():
            #     self.data_queue.get(block=False)

        

if __name__ == '__main__':
    rospy.init_node("panda_controller", log_level=rospy.WARN)

    panda_controller = PandaController()
    # panda_controller.start_processing()

    # signal.signal(signal.SIGINT, panda_controller.stop_processing)

    panda_controller.process()
    # while not rospy.is_shutdown():
    #     rospy.spin()