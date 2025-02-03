#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from object_detection import ObjectDetector  # Classe per il rilevamento della box (definita separatamente)

class ObjectRecognition:
    def __init__(self):
        # Inizializza il bridge per la conversione delle immagini
        self.bridge = CvBridge()

        # Subscriber per le immagini RGB e di profondità
        self.rgb_sub = Subscriber("/kinect/color/image_raw", Image)
        self.depth_sub = Subscriber("/kinect/depth/image_raw", Image)

        # Sincronizzazione dei topic
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        # Inizializza l'oggetto di rilevamento
        self.object_detector = ObjectDetector()

        # Publisher per il movimento del robot
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def callback(self, rgb_data, depth_data):
        try:
            # Converti le immagini ROS in formato OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, rgb_data.encoding)
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding=depth_data.encoding)

            # Rilevamento della box nelle immagini RGB
            boxes = self.object_detector.detect_box(rgb_image)
            if boxes:
                for box in boxes:
                    # Calcola la posizione 3D della box
                    x, y, z = self.get_3d_coordinates(box, depth_image)
                    rospy.loginfo(f"Object detected at position: x={x}, y={y}, z={z}")

                    # Comando al robot per muoversi verso la box
                    self.move_robot(x, y, z)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def get_3d_coordinates(self, rgb_box, depth_image):
        # Calcola la posizione 3D della box usando i dati di profondità
        x_center = rgb_box[0]
        y_center = rgb_box[1]
        
        depth_value = depth_image[y_center, x_center]  # Ottieni la profondità al centro della box
        z = depth_value
        x = (x_center - 320) * z / 525  # Calibrazione della fotocamera (esempio)
        y = (y_center - 240) * z / 525  # Calibrazione della fotocamera (esempio)
        
        return (x, y, z)

    def move_robot(self, x, y, z):
        # Comando di movimento verso la box (adatta questo metodo al tuo controllo robot)
        move_cmd = Twist()
        move_cmd.linear.x = x  # Muovi in avanti
        move_cmd.linear.y = y  # Muovi lateralmente
        move_cmd.linear.z = z  # Muovi verticalmente

        self.cmd_pub.publish(move_cmd)

if __name__ == '__main__':
    rospy.init_node('object_recognition', anonymous=True)
    obj_rec = ObjectRecognition()
    rospy.spin()
