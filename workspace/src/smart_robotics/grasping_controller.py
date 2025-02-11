#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose
from panda_robot import PandaArm

class GraspingController:
    def __init__(self):
        self.panda = PandaArm()
        self.gripper = self.panda.get_gripper()
        self.panda.untuck()
        self.gripper.open()

        # Subscriber al topic che contiene la posizione dell'oggetto
        rospy.Subscriber('/detected_object_pose', Pose, self.pose_callback)

    def pose_callback(self, pose):
        # Ottieni la posizione dall'oggetto rilevato
        target_pos = [pose.position.x, pose.position.y, pose.position.z]
        target_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        
        # Muovi il robot verso la posizione dell'oggetto
        self.move_robot(target_pos, target_quat)

    def move_robot(self, target_pos, target_quat):
        # Muovi il robot verso la posizione dell'oggetto
        ret, tgt_joints = self.panda.inverse_kinematics(target_pos, target_quat)
        if ret:
            self.panda.move_to_joint_position(tgt_joints)
            rospy.sleep(1)

        # Chiudi il gripper per afferrare l'oggetto
        self.gripper.close()
        rospy.sleep(2)

        print("Oggetto afferrato con successo!")

if __name__ == '__main__':
    rospy.init_node('grasping_controller', anonymous=True)
    controller = GraspingController()
    rospy.spin()
