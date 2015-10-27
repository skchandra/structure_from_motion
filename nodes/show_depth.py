#!/usr/bin/env python

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

# D = np.array( [0.12969982463039115, -0.3662323365747535, -0.0047588179431673535, 0.0032280633422334627, 0.0])
# K = np.array( [[672.9140718714626, 0.0, 317.4277026418419], [0.0, 670.9812424489073, 216.90441559892045], [0.0, 0.0, 1.0]])
# W = np.array([[0.0, -1.0, 0.0],
# 			  [1.0, 0.0, 0.0],
# 			  [0.0, 0.0, 1.0]])

D = np.array( [0.08273, -0.11202, 0.00054, -0.00939, 0.00000])
K = np.array( [[669.26503, 0.00000, 297.04905], 
				[0.00000, 669.34471, 235.20167], 
				[0.00000, 0.00000, 1.00000]]) 
W = np.array([[0.0, -1.0, 0.0],
			  [1.0, 0.0, 0.0],
			  [0.0, 0.0, 1.0]])

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
pt_num = 0
im1_pts = []
im2_pts = []

def triangulate_points(pt_set1, pt_set2, P, P1):
	my_points = cv2.triangulatePoints(P,P1,pt_set1.T,pt_set2.T)
	projected_points_1 = P.dot(my_points)
	
	# convert to inhomogeneous coordinates
	for i in range(projected_points_1.shape[1]):
		projected_points_1[0,i] /= projected_points_1[2,i]
		projected_points_1[1,i] /= projected_points_1[2,i]
		projected_points_1[2,i] /= projected_points_1[2,i]

	projected_points_2 = P1.dot(my_points)
	# convert to inhomogeneous coordinates
	for i in range(projected_points_2.shape[1]):
		projected_points_2[0,i] /= projected_points_2[2,i]
		projected_points_2[1,i] /= projected_points_2[2,i]
		projected_points_2[2,i] /= projected_points_2[2,i]

	# convert to inhomogeneous coordinates
	for i in range(projected_points_2.shape[1]):
		my_points[0,i] /= my_points[3,i]
		my_points[1,i] /= my_points[3,i]
		my_points[2,i] /= my_points[3,i]
		my_points[3,i] /= my_points[3,i]

	return my_points.T

def test_epipolar(E,pt1,pt2):
	pt1_h = np.zeros((3,1))
	pt2_h = np.zeros((3,1))
	pt1_h[0:2,0] = pt1.T
	pt2_h[0:2,0] = pt2.T
	pt1_h[2] = 1.0
	pt2_h[2] = 1.0
	return pt2_h.T.dot(E).dot(pt1_h)

def test_triangulation(P,pcloud):
	P4x4 = np.eye(4)
	P4x4[0:3,:] = P
	pcloud_3d = pcloud[:,0:3]
	projected = cv2.perspectiveTransform(np.array([pcloud_3d]),P4x4)
	return np.mean(projected[0,:,2]>0.0)

def mouse_event(event,x,y,flag,im):
	if event == cv2.EVENT_FLAG_LBUTTON:
		if x < im.shape[1]/2.0:
			l = F.dot(np.array([x,y,1.0]))
			m = -l[0]/l[1]
			b = -l[2]/l[1]
			# equation of the line is y = m*x+b
			y_for_x_min = m*0.0+b
			y_for_x_max = m*(im.shape[1]/2.0-1)+b
			# plot the epipolar line
			cv2.line(im,(int(im.shape[1]/2.0),int(y_for_x_min)),(int(im.shape[1]-1.0),int(y_for_x_max)),(255,0,0))

def show_depth(im1, im2, correspondences):
	im1_pts = np.zeros((len(correspondences[0]),2))
	im2_pts = np.zeros((len(correspondences[1]),2))

	im = np.array(np.hstack((im1,im2)))

	# plot the points
	for i in range(len(im1_pts)):
		im1_pts[i,0] = correspondences[0][i][0]
		im1_pts[i,1] = correspondences[0][i][1]
		im2_pts[i,0] = correspondences[1][i][0]
		im2_pts[i,1] = correspondences[1][i][1]

		# cv2.circle(im,(int(im1_pts[i,0]),int(im1_pts[i,1])),2,(255,0,0),2)
		# cv2.circle(im,(int(im2_pts[i,0]+im1.shape[1]),int(im2_pts[i,1])),2,(255,0,0),2)

	im1_pts_augmented = np.zeros((1,im1_pts.shape[0],im1_pts.shape[1]))
	im1_pts_augmented[0,:,:] = im1_pts
	im2_pts_augmented = np.zeros((1,im2_pts.shape[0],im2_pts.shape[1]))
	im2_pts_augmented[0,:,:] = im2_pts

	#im1_pts_ud = im1_pts_augmented
	#im2_pts_ud = im2_pts_augmented
	print im1_pts_augmented.shape
	print im2_pts_augmented.shape
	im1_pts_ud = cv2.undistortPoints(im1_pts_augmented,K,D)
	im2_pts_ud = cv2.undistortPoints(im2_pts_augmented,K,D)

	E, mask = cv2.findFundamentalMat(im1_pts_ud,im2_pts_ud,cv2.FM_RANSAC)

	im1_pts_ud_fixed, im2_pts_ud_fixed = cv2.correctMatches(E, im1_pts_ud, im2_pts_ud)
	use_corrected_matches = True
	if not(use_corrected_matches):
		im1_pts_ud_fixed = im1_pts_ud
		im2_pts_ud_fixed = im2_pts_ud

	epipolar_error = np.zeros((im1_pts_ud_fixed.shape[1],))
	for i in range(im1_pts_ud_fixed.shape[1]):
		epipolar_error[i] = test_epipolar(E,im1_pts_ud_fixed[0,i,:],im2_pts_ud_fixed[0,i,:])

	#F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
	F = K.T * E * K
	U, Sigma, V = np.linalg.svd(E)

	R1 = U * W * V.T
	W_inv = np.linalg.inv(W)
	R2 = U * W_inv * V.T
	#R1 = U.dot(W).dot(V)
	#R2 = U.dot(W.T).dot(V)

	if np.linalg.det(R1)+1.0 < 10**-8:
		# flip sign of E and recompute everything
		E = -E
		F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
		U, Sigma, V = np.linalg.svd(E)

		R1 = U.dot(W).dot(V)
		R2 = U.dot(W.T).dot(V)

	t1 = U[:,2]
	t2 = -U[:,2]

	P = np.array([[1.0,	0.0, 0.0, 0.0],
				  [0.0,	1.0, 0.0, 0.0],
				  [0.0,	0.0, 1.0, 0.0]]);
	P1_possibilities = []
	P1_possibilities.append(np.column_stack((R1,t1)))
	P1_possibilities.append(np.column_stack((R1,t2)))
	P1_possibilities.append(np.column_stack((R2,t1)))
	P1_possibilities.append(np.column_stack((R2,t2)))

	pclouds = []
	for P1 in P1_possibilities:
		pclouds.append(triangulate_points(im1_pts_ud_fixed, im2_pts_ud_fixed, P, P1))

	infront_of_camera = []
	for i in range(len(P1_possibilities)):
		infront_of_camera.append(test_triangulation(P,pclouds[i])+test_triangulation(P1_possibilities[i],pclouds[i]))

	best_pcloud = pclouds[np.argmax(infront_of_camera)]
	# depths = best_pcloud[:,2] - min(best_pcloud[:,2])
	# depths = depths / max(depths)

	x = [i[0] for i in best_pcloud]
	y = [i[1] for i in best_pcloud]
	z = [i[2] for i in best_pcloud]
	z = (z - min(z)) / max(z)
	#z = np.zeros(len(x))

	m = 3
	for i in range(len(x)):
		if abs(x[i] - np.mean(x)) > m * np.std(x) or abs(y[i] - np.mean(y)) > m * np.std(y):
			x[i] = 0
			y[i] = 0
			z[i] = 0

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z)
	plt.show()
	

	for i in range(best_pcloud.shape[0]):
		cv2.circle(im,(int(im1_pts[i,1]),int(im1_pts[i,0])),1,(0,255,0),1)

	cv2.imshow("MYWIN",im)
	cv2.setMouseCallback("MYWIN",mouse_event,im)
	while True:
		cv2.imshow("MYWIN",im)
		cv2.waitKey(50)
	cv2.destroyAllWindows()