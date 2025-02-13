
import threading
import rospy, tf, rospkg, random
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetModelState
from geometry_msgs.msg import Quaternion, Pose, Point
from std_srvs.srv import Empty

class ObjectSpawnerController():
    
    def __init__(self) -> None:
        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path('smart_robotics')+"/urdf/"
        self.objects = []
        self.objects.append(self.path+"red_cube.urdf")
        self.col = 0        
        self.spawned_objects = []
        
        self.is_spawning = threading.Event()
        self.rate = rospy.Rate(0.3)
        
        self.sm = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
        self.dm = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.ms = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)


    def checkModel(self):
        res = self.ms("cube", "world")
        return res.success

    def getPosition(self):
        res = self.ms("cube", "world")
        return res.pose.position.z

    def spawnModel(self):
        sampling_idx = 0 # TODO: get random object index
        object = self.objects[sampling_idx]
        with open(object,"r") as f:
            object_urdf = f.read() #TODO: Make it parametrizable in terms of scale and color

        spawn_idx = len(self.spawned_objects)
        quat = tf.transformations.quaternion_from_euler(0,0,0) #TODO: Give some randomness to rotation (in a defined range)
        orient = Quaternion(quat[0],quat[1],quat[2],quat[3])
        pose = Pose(Point(x=0.5,y=-0.8,z=0.75), orient) #TODO: Given some randomness to position (in a defined range)
        object_id = f"object_{spawn_idx}"
        self.sm(object_id, object_urdf, '', pose, 'world')
        self.spawned_objects.append(object_id)
        # rospy.sleep(1)

    def deleteModel(self):
        self.dm("cube")
        rospy.sleep(1)

    def shutdown_hook(self):
        self.deleteModel()
        print("Shutting down")


    def process(self):
        while not rospy.is_shutdown():
            self.spawnModel()
            self.rate.sleep()

from gazebo_conveyor.srv import ConveyorBeltControl

if __name__ == '__main__':
    rospy.init_node("spawner_controller")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/spawn_urdf_model")
    rospy.wait_for_service("/gazebo/get_model_state")
    rospy.wait_for_service('/conveyor/control')

    conveyor_control = rospy.ServiceProxy('/conveyor/control', ConveyorBeltControl) # This will be moved in another file
    response = conveyor_control(power=15.0)

    rospy.on_shutdown(conveyor_control.close) # Need to call delete model

    spawner_controller = ObjectSpawnerController()
    spawner_controller.process()
    