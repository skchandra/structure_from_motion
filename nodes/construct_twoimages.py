#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

from sfm import *

class Reconstruct3D:
	def __init__(self):
		"""Initialize subscriptions to topics, such as the raw video"""

		rospy.init_node('construct_images')
		rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

		# Raw images from the Neato
		self.image = None
		self.bridge = CvBridge()

		# Stores 2 images when the 'c' key is pressed
		self.image1 = None
		self.image2 = None

		self.calculated_flow = False

	def image_callback(self, msg):
		"""Callback function to continuously set self.image"""

		self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

	def run(self):
		"""Runs the program until q pressed or program stopped. In this process the two images are saved 
		and the function to calculate the optical flow and depth is called."""

		r = rospy.Rate(10)

		while not rospy.is_shutdown():
			#ret, self.image = self.cap.read()
			if self.image != None:
				if self.image2 == None:
					cv2.imshow('image_raw', self.image)
					if cv2.waitKey(1) & 0xFF == ord('c'):	#if 'c' key is pressed take a pic from Neato camera
						if self.image1 == None:
							self.image1 = self.image
						elif self.image2 == None:
							self.image2 = self.image
				elif not self.calculated_flow:	#if the two images are captured and optical flow hasn't been calculated, set to True and run calculations
					self.calculated_flow = True
					construct(self.image1, self.image2) 	#call function from sfm.py file 

			if cv2.waitKey(30) & 0xFF == ord('q'):
				cv2.destroyAllWindows()

			r.sleep()

if __name__ == '__main__':
	robot = Reconstruct3D()
	robot.run()