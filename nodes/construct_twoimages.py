#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import cv2, numpy as np, math, matplotlib.pyplot as plt
from cv_bridge import CvBridge

from show_depth import *

class Reconstruct3D:
	def __init__(self):
		rospy.init_node('construct_images')
		rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
		rospy.Subscriber('/odom', Odometry, self.odom_callback)

		self.cap = cv2.VideoCapture(0)

		self.image = None
		self.bridge = CvBridge()

		self.image1 = None
		self.image2 = None

		self.correspondences = ([], [])

		self.calculated_flow = False

	def image_callback(self, msg):
		ret, self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


	def odom_callback(self, msg):
		pass

	def optical_flow(self):
		sift = cv2.SIFT()

		old_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
		new_image = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

		pts0, des0 = sift.detectAndCompute(old_image, None)
		pts1, des1 = sift.detectAndCompute(new_image, None)

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des0, des1, k=2)

		good = []
		for m,n in matches:
		    if m.distance < 50:
		        good.append([m])


		# # Apply ratio test
		# for m,n in matches:
		#     if m.distance < 0.75*n.distance:
		#     	self.correspondences[0].append((m.queryIdx, m.trainIdx))
		#     	self.correspondences[1].append((n.queryIdx, n.trainIdx))

		# sift = cv2.SIFT()

		# pts0 = sift.detect(self.image1, None)
		# for i in range(len(pts0)):
		# 	pts0[i] = [[pts0[i].pt[0], pts0[i].pt[1]]]
		# pts0 = np.array(pts0, np.float32)

		# pts1 = np.zeros(pts0.shape)
		# for i in range(len(pts0)):
		# 	y = int(pts0[i][0][1] + 0.5)
		# 	x = int(pts0[i][0][0] + 0.5)
		# 	dy = flow[y][x][0]
		# 	dx = flow[y][x][1]
		# 	self.correspondences[0].append((y,x))
		# 	self.correspondences[1].append((y+dy, x+dx))


		# flow = cv2.calcOpticalFlowFarneback(old_image, new_image, 0.5, 3, 15, 3, 5, 1.2, 0)

		# for y in range(old_image.shape[0]):
		# 	for x in range(old_image.shape[1]):
		# 		if math.sqrt( flow[y][x][0]**2 + flow[y][x][1]**2 > 70 ):
		# 			self.correspondences[0].append((y,x))
		# 			self.correspondences[1].append((y + flow[y][x][0],
		# 											x + flow[y][x][1]))

		# lk_params = dict( winSize = (10,10),
		# 				  maxLevel = 10,
		# 				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) )

		# color = [255,0,0]

		# # Keypoint matching
		# old_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
		# new_image = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
		# sift = cv2.SIFT()

		# pts0 = sift.detect(self.image1, None)
		# for i in range(len(pts0)):
		# 	pts0[i] = [[pts0[i].pt[0], pts0[i].pt[1]]]
		# pts0 = np.array(pts0, np.float32)
		# pts1, st, err = cv2.calcOpticalFlowPyrLK(old_image, new_image, pts0, None, **lk_params)

		# good_new = pts1[st==1]
		# good_old = pts0[st==1]

		# for i,(new,old) in enumerate(zip(good_new, good_old)):
		# 	a,b = new.ravel()
		# 	c,d = old.ravel()
		# 	self.correspondences[0].append((c,d))
		# 	self.correspondences[1].append((a,b))
		# 	# cv2.line(self.image2, (a,b), (c,d), color, 2)
		# 	# cv2.circle(self.image2, (a,b), 5, color, -1)
		
	def run(self):
		r = rospy.Rate(10)

		while not rospy.is_shutdown():
			ret, self.image = self.cap.read()
			if self.image != None:
				if self.image2 == None:
					cv2.imshow('image_raw', self.image)
					if cv2.waitKey(20) & 0xFF == ord('c'):
						if self.image1 == None:
							self.image1 = self.image
						elif self.image2 == None:
							self.image2 = self.image
				elif not self.calculated_flow:
					self.optical_flow()
					self.calculated_flow = True
					show_depth(self.image1, self.image2, self.correspondences)

					#cv2.imshow('before flow', self.image1)
					#cv2.imshow('optical flow', self.image2)

			if cv2.waitKey(30) & 0xFF == ord('q'):
				cv2.destroyAllWindows()

			r.sleep()

if __name__ == '__main__':
	robot = Reconstruct3D()
	robot.run()