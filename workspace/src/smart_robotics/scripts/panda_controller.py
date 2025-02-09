import math
from multiprocessing import Queue
from queue import Full
import signal
import threading
from time import sleep

import numpy as np
import rospy
from panda_robot import PandaArm
from custom_msg.msg import PosesWithScales


def calculate_rotation(start, target):
    delta_x = target[0] - start[0]
    delta_y = target[1] - start[1]
    
    angle_rad = math.atan2(delta_y, delta_x)
    
    return angle_rad


class PandaController:
    def __init__(self):
        

        self.data_queue = Queue(maxsize=1)

        self.panda = PandaArm()
        self.gripper = self.panda.get_gripper()
        self.gripper.open()
        self.panda.untuck()

        self.bins = {
            'paper': np.array([-0.377553, -0.517819, 0]),
            'glass': np.array([-0.377553, 0.517819, 0])
        }
        
        self.is_processing = threading.Event()
        self.is_processing.set()
        self.processing_thread = threading.Thread(target=self._process)

        self.objects_poses_subscriber = rospy.Subscriber("/kinect_controller/detected_poses", PosesWithScales, self.poses_callback)
        self.processing_thread.start()


    def poses_callback(self, msg):
        pos = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses])
        orientations = np.array([[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] for pose in msg.poses])
        scales = np.array([[scale.x, scale.y, scale.z] for scale in msg.scales])

        if not self.data_queue.full():
            self.data_queue.put({'pos': pos, 'orients': orientations, 'scales': scales}, block=False, timeout=0.01)
        else:
            pass # Don't do anything we skip these detections.

    def grasp_task(self, target_pos, target_quat = np.array([1., 0., 0., 0.])):
        target_pos[-1] += 0.11

        intermediate_pose = target_pos.copy()
        intermediate_pose[-1] += 0.2

        ret, int_joints = self.panda.inverse_kinematics(intermediate_pose, target_quat)
        if ret or True:
            self.panda.move_to_joint_position(int_joints)
        
        target_pos[0] += 0.01
        ret, tgt_joints = self.panda.inverse_kinematics(target_pos, target_quat)
        if ret or True:
            self.panda.move_to_joint_position(tgt_joints)
        
        rospy.sleep(0.01)

        object_width = 0.05 
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

        rospy.sleep(1)

        x = 4
    
    def _process(self):
        while not rospy.is_shutdown() and self.is_processing.is_set():
            if self.data_queue.empty():
                sleep(0.01)
                continue
            
            data = self.data_queue.get(block=False)
            
            if data is None:
                sleep(0.01)
                continue

            positions = data['pos']
            scales = data['scales']
            orients = data['orients']
            N_dets = positions.shape[0]
            if N_dets == 0:
                sleep(0.01)
                continue
                
            assert N_dets == scales.shape[0] == orients.shape[0], 'Shapes not consisntents.'

            for i in range(N_dets):
                # Do all the task here
                self.grasp_task(positions[i], orients[i])

                bin_key = 'paper' if i%2 == 0 else 'glass'
                tgt_bin = self.bins[bin_key]
                self.release_task(tgt_bin)

            # Clean the queue from old predictions
            if self.data_queue.full():
                self.data_queue.get(block=False)

        

if __name__ == '__main__':
    rospy.init_node("panda_controller", log_level=rospy.WARN)

    panda_controller = PandaController()
    # panda_controller.start_processing()

    # signal.signal(signal.SIGINT, panda_controller.stop_processing)

    while not rospy.is_shutdown():
        rospy.spin()