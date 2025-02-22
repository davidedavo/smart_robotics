#!/usr/bin/env python


from multiprocessing import Queue
import threading
from time import sleep
import numpy as np
import rospy
from gazebo_conveyor.srv import ConveyorBeltControl
from std_msgs.msg import String
from custom_msg.msg import Detection3D
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
        self.is_conveyor_on = threading.Event()

        self.rate = rospy.Rate(30)
        self.robot_status = 'idle'

        self.bins = {
            0: np.array([-0.377553, -0.517819, 0]),
            1: np.array([-0.377553, 0.517819, 0])
        }
        
        self.objects_poses_subscriber = rospy.Subscriber("/kinect_controller/detected_poses", Detection3D, self.poses_callback)
        self.panda_status_subscriber = rospy.Subscriber('/panda/pick_status', String, self.panda_status_callback)
        self.start_factory()

    
    def poses_callback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        orientation = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        scale = np.array([msg.scale.x, msg.scale.y, msg.scale.z])
        class_id = msg.class_id

        with self.data_lock:
            if self.data_queue.full():
                self.data_queue.get(block=False, timeout=0.01)
            self.data_queue.put({'pos': pos, 'orient': orientation, 'scale': scale, 'class_id': class_id}, block=False, timeout=0.01)


    def panda_status_callback(self, msg):
        status = msg.data
        rospy.loginfo(f'Robot status: {status}')
        self.robot_status = status
        if status != 'grasping' and not self.is_conveyor_on.is_set():
            self.start_factory()
 
    
    def start_factory(self):
        rospy.loginfo('Starting Factory')
        self.is_picking.clear()
        self.conveyor_control(power=10.0)
        self.is_conveyor_on.set()
        sleep(4.0)
        if self.is_conveyor_on.is_set():
            self.spawn_service(True)


    def stop_factory(self):
        rospy.loginfo('Stop Factory')
        self.is_picking.set()
        self.is_conveyor_on.clear()
        self.conveyor_control(power=0.0)
        self.spawn_service(False)
    

    def shutdown_hook(self):
        self.stop_factory()
        self.conveyor_control.close()
        self.spawn_service.close()

    def send_pick_and_place_req(self, position, orient, scale, place_pos):
        px, py, pz = position[0], position[1], position[2]
        qw, qx, qy, qz = orient[0], orient[1], orient[2], orient[3]
        sx, sy, sz = scale[0], scale[1], scale[2]

        pose = Pose(position=Point(px, py, pz), orientation=Quaternion(qx, qy, qz, qw))
        scale = Vector3(sx, sy, sz)
        place_pos = Vector3(place_pos[0], place_pos[1], place_pos[2])
        self.pick_place_server(pose=pose, scale=scale, place_pos=place_pos)


    def process(self):
        while not rospy.is_shutdown():            
            data = None
            with self.data_lock:
                if not self.data_queue.empty():
                    data = self.data_queue.get(block=False)
            
            if self.is_picking.is_set() or data is None:
                self.rate.sleep()
                continue
            
            position = data['pos']
            scale = data['scale']
            orient = data['orient']
            class_id = data['class_id']

            distance = np.abs(position[1] - 0)
            is_pick_position =  distance < 1e-2

            # print(positions)

            if is_pick_position and class_id in [0, 1]:
                self.is_picking.set()
                self.stop_factory()
                pick_pos = position
                pick_scale = scale
                pick_orient = orient
                place_pos = self.bins[class_id]
                
                self.send_pick_and_place_req(pick_pos, pick_orient, pick_scale, place_pos)
                
            self.rate.sleep()




if __name__ == '__main__':
    rospy.init_node("factory_controller")

    factory_controller = FactoryController()
    
    rospy.on_shutdown(factory_controller.shutdown_hook)
    factory_controller.process()