#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

class Reconstruct3D:
	def __init__(self):
		rospy.init_node('construct_images')
		rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
		rospy.Subscriber('/odom', Odometry, self.odom_callback)

		self.image = None
		self.bridge = CvBridge()

	def image_callback(self, msg):
		self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


	def odom_callback(self, msg):
		pass

	def run(self):
		r = rospy.Rate(10)
		while not rospy.is_shutdown():
			if self.image != None:
				cv2.imshow('image_raw', self.image)
				cv2.waitKey(20)
			r.sleep()

if __name__ == '__main__':
	robot = Reconstruct3D()
	robot.run()