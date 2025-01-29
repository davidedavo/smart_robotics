import sys

import numpy as np
import rospy
from panda_robot import PandaArm
from geometry_msgs.msg import Pose


# Imposta i parametri per la larghezza dell'oggetto e la forza
object_width = 0.05  # Supponendo che la box abbia una larghezza di 5 cm
grasp_force = 20.0   # Forza di presa in Newton
grasp_speed = 0.05   # Velocità di chiusura del gripper in m/s



def main():
    rospy.init_node('panda_sim')

    panda = PandaArm()
    gripper = panda.get_gripper()
    panda.untuck()
    gripper.open()

    target_pose = np.array([0.45645, -0.107404, 0.426083])
    target_quat = np.array([1., 0., 0., 0.])

    target_pose[-1] += 0.1

    intermediate_pose = target_pose.copy()
    intermediate_pose[-1] += 0.2
    
    ret, int_joints = panda.inverse_kinematics(intermediate_pose, target_quat)
    if ret or True:
        panda.move_to_joint_position(int_joints)

    rospy.sleep(1)
    
    ret, tgt_joints = panda.inverse_kinematics(target_pose, target_quat)
    if ret or True:
        panda.move_to_joint_position(tgt_joints)
    
    rospy.sleep(2)

    # Close the gripper to grasp the unit cube
    #gripper.close()
    #gripper.apply_force(10.0)
    # Chiudi il gripper e afferra l'oggetto
    success = gripper.grasp(width=object_width, force=grasp_force, speed=grasp_speed)

    if success:
        print("Oggetto afferrato con successo!")
    else:
        print("Impossibile afferrare l'oggetto.")

    rospy.sleep(2)

    panda.move_to_joint_position(int_joints)   

    rospy.spin()

if __name__ == '__main__':
    main()