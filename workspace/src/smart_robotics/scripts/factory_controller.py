#!/usr/bin/env python


from multiprocessing import Queue
import threading
from time import sleep
import numpy as np
import rospy
from gazebo_conveyor.srv import ConveyorBeltControl
from custom_msg.msg import PosesWithScales
from custom_msg.srv import SpawnObject, SpawnObjectResponse, PickObject
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3


class FactoryController:
    def __init__(self):
        rospy.wait_for_service('/conveyor/control')
        rospy.wait_for_service('/spawner/spawn_objects')
        rospy.wait_for_service('/spawner/spawn_objects')
        rospy.wait_for_service('/panda/pick_and_place')
        self.conveyor_control = rospy.ServiceProxy('/conveyor/control', ConveyorBeltControl)
        self.spawn_service = rospy.ServiceProxy('/spawner/spawn_objects', SpawnObject)
        self.pick_place_server = rospy.ServiceProxy('/panda/pick_and_place', PickObject)
        
        self.data_queue = Queue(maxsize=1)
        self.data_lock = threading.Lock()
        self.is_picking = threading.Event()

        self.rate = rospy.Rate(30)
        
        self.objects_poses_subscriber = rospy.Subscriber("/kinect_controller/detected_poses", PosesWithScales, self.poses_callback)
        self.start_factory()

    
    
    def poses_callback(self, msg):
        pos = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses])
        orientations = np.array([[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z] for pose in msg.poses])
        scales = np.array([[scale.x, scale.y, scale.z] for scale in msg.scales])

        with self.data_lock:
            if self.data_queue.full():
                self.data_queue.get(block=False, timeout=0.01)
            self.data_queue.put({'pos': pos, 'orients': orientations, 'scales': scales}, block=False, timeout=0.01)
 
    
    def start_factory(self):
        self.conveyor_control(power=10.0)
        self.spawn_service(True)

    def stop_factory(self):
        self.conveyor_control(power=0.0)
        self.spawn_service(False)
    
    def shutdown_hook(self):
        self.stop_factory()
        self.conveyor_control.close()
        self.spawn_service.close()

    def send_pick_and_place_req(self, position, orient, scale):
        px, py, pz = position[0], position[1], position[2]
        qw, qx, qy, qz = orient[0], orient[1], orient[2], orient[3]
        sx, sy, sz = scale[0], scale[1], scale[2]

        pose = Pose(position=Point(px, py, pz), orientation=Quaternion(qx, qy, qz, qw))
        scale = Vector3(sx, sy, sz)
        self.pick_place_server(pose=pose, scale=scale)


    def process(self):
        while not rospy.is_shutdown():            
            data = None
            with self.data_lock:
                if not self.data_queue.empty():
                    data = self.data_queue.get(block=False)
            
            if self.is_picking.is_set() or data is None:
                self.rate.sleep()
                continue
            
            positions = data['pos']
            scales = data['scales']
            orients = data['orients']
            N_dets = positions.shape[0]
            if N_dets == 0:
                self.rate.sleep()
                continue
                
            assert N_dets == scales.shape[0] == orients.shape[0], 'Shapes not consisntents.'

            distances = np.abs(positions[:, 1] - 0)
            is_pick_position =  distances < 1e-2

            print(positions)

            if is_pick_position.any():
                self.is_picking.set()
                self.stop_factory()
                pick_pos = positions[is_pick_position][0]
                pick_scale = scales[is_pick_position][0]
                pick_orient = orients[is_pick_position][0]
                self.send_pick_and_place_req(pick_pos, pick_orient, pick_scale)
                pass

            self.rate.sleep()




if __name__ == '__main__':
    rospy.init_node("factory_controller")

    factory_controller = FactoryController()
    
    rospy.on_shutdown(factory_controller.shutdown_hook)
    factory_controller.process()