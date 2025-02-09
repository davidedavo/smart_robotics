from multiprocessing import Queue
from queue import Full
import signal
import threading
from time import sleep

import numpy as np
import rospy
from panda_robot import PandaArm
from custom_msg.msg import PosesWithScales


class PandaController:
    def __init__(self):
        self.objects_poses_subscriber = rospy.Subscriber("/kinect_controller/detected_poses", PosesWithScales, self.poses_callback)

        self.data_queue = Queue(maxsize=1)

        self.panda = PandaArm()
        self.gripper = self.panda.get_gripper()
        self.panda.untuck()
        self.gripper.open()
        
        self.is_processing = threading.Event()
        self.is_processing.set()
        self.processing_thread = threading.Thread(target=self._process)

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

        rospy.sleep(1)
    
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

                self.panda.untuck()

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