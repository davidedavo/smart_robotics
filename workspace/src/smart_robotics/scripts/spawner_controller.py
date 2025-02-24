#!/usr/bin/env python

import threading
import rospy, tf, rospkg, random
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetModelState
from geometry_msgs.msg import Quaternion, Pose, Point
from custom_msg.srv import SpawnObject, SpawnObjectResponse

class ObjectSpawnerController():
    def __init__(self) -> None:
        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path('smart_robotics')+"/urdf/"
        self.objects = [
            self.path+"red_cube.urdf",
            self.path+"red_rectangle.urdf",
            self.path+"red_cylinder.urdf"
        ]
        self.col = 0        
        self.spawned_objects = []
        
        self.is_spawning = threading.Event()
        self.restart_spawning = threading.Event()
        self.rate = rospy.Rate(0.3)

        self.spawn_service = rospy.Service('/spawner/spawn_objects', SpawnObject, self.handle_spawn_service)
        
        self.sm = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
        self.dm = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.ms = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)


    def handle_spawn_service(self, req):
        if req.spawn:
            rospy.loginfo("Start spawning objects.")
            self.is_spawning.set()
        else:
            rospy.loginfo("Stop spawning objects.")
            self.is_spawning.clear()
        return SpawnObjectResponse(True)


    def checkModel(self):
        res = self.ms("cube", "world")
        return res.success

    def getPosition(self):
        res = self.ms("cube", "world")
        return res.pose.position.z

    def spawnModel(self):
        sampling_idx = random.randint(0, len(self.objects)-1)
        object = self.objects[sampling_idx]
        with open(object,"r") as f:
            object_urdf = f.read() #TODO: Make it parametrizable in terms of scale and color


        scale = round(random.uniform(0.6, 1.0), 2)  # Scala casuale --> 0.57 scivola
        if "red_cube" in object_urdf:
            new_size = 0.06 * scale
            object_urdf = object_urdf.replace('<box size="0.06 0.06 0.06"/>',
                                            f'<box size="{new_size} {new_size} {new_size}"/>')
            
        elif "red_rectangle" in object_urdf:
            new_size_1 = 0.12 * scale
            new_size_2 = 0.06 * scale
            object_urdf = object_urdf.replace('<box size="0.12 0.06 0.06"/>',
                                            f'<box size="{new_size_1} {new_size_2} {new_size_2}"/>')
                
        elif "red_cylinder" in object_urdf:
            new_radius = 0.03 * scale
            new_length = 0.1 * scale
            object_urdf = object_urdf.replace('<cylinder radius="0.03" length="0.1"/>',
                                            f'<cylinder radius="{new_radius}" length="{new_length}"/>')
                

        spawn_idx = len(self.spawned_objects)
        quat = tf.transformations.quaternion_from_euler(0,0,0) #TODO: Give some randomness to rotation (in a defined range)
        orient = Quaternion(quat[0],quat[1],quat[2],quat[3])
        pose = Pose(Point(x=0.5,y=-0.8,z=0.75), orient) #TODO: Given some randomness to position (in a defined range)
        object_id = f"object_{spawn_idx}"
        print(f"\n#####\noggetto --> {object_id}    -----    scale --> {scale}\n#######\n")
        self.sm(object_id, object_urdf, '', pose, 'world')
        self.spawned_objects.append(object_id)
        # rospy.sleep(1)

    def deleteModel(self):
        self.dm("cube")
        rospy.sleep(1)

    def shutdown_hook(self):
        # TODO: Need to delete all models
        self.deleteModel()
        print("Shutting down")


    def process(self):
        while not rospy.is_shutdown():
            if self.is_spawning.is_set():
                self.spawnModel()
                # self.is_spawning.clear()
            self.rate.sleep()

from gazebo_conveyor.srv import ConveyorBeltControl

if __name__ == '__main__':
    rospy.init_node("spawner_controller")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/spawn_urdf_model")
    rospy.wait_for_service("/gazebo/get_model_state")

    spawner_controller = ObjectSpawnerController()
    rospy.on_shutdown(spawner_controller.shutdown_hook) 
    spawner_controller.process()
    