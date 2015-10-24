import cv2
import numpy as np 

D = np.array( [0.12969982463039115, -0.3662323365747535, -0.0047588179431673535, 0.0032280633422334627, 0.0])
K = np.array( [[672.9140718714626, 0.0, 317.4277026418419], [0.0, 670.9812424489073, 216.90441559892045], [0.0, 0.0, 1.0]])
W = np.array([[0.0, -1.0, 0.0],
			  [1.0, 0.0, 0.0],
			  [0.0, 0.0, 1.0]])

def sfm(im1, im2):
	old_image = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	new_image = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(old_image, new_image, 0.5, 3, 12, 3, 5, 1.2, 0)
	correspondences = ([], [])

	for y in range(0, old_image.shape[0], 6):
			for x in range(0, old_image.shape[1], 6):
				if abs(flow[y][x][0] > 0.1) and abs(flow[y][x][1] > 0.1):
					correspondences[0].append((y,x))
					correspondences[1].append((y + flow[y][x][0],
													x + flow[y][x][1]))

	correspondences[0] = cv2.undistortPoints(correspondences[0], K, D)
	correspondences[1] = cv2.undistortPoints(correspondences[1], K, D)

	# E, mask = cv2.findFundamentalMat(correspondences[0], correspondences[1], cv2.FM_RANSAC)
	
	# F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
	# U, Sigma, V = np.linalg.svd(E)

	# R1 = U.dot(W).dot(V)
	# R2 = U.dot(W.T).dot(V)

	# if np.linalg.det(R1)+1.0 < 10**-8:
	# 	# flip sign of E and recompute everything
	# 	E = -E
	# 	F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
	# 	U, Sigma, V = np.linalg.svd(E)

	# 	R1 = U.dot(W).dot(V)
	# 	R2 = U.dot(W.T).dot(V)

	# t1 = U[:,2]
	# t2 = -U[:,2]

	# P = np.array([[1.0,	0.0, 0.0, 0.0],
	# 			  [0.0,	1.0, 0.0, 0.0],
	# 			  [0.0,	0.0, 1.0, 0.0]]);
	# P1_possibilities = []
	# P1_possibilities.append(np.column_stack((R1,t1)))
	# P1_possibilities.append(np.column_stack((R1,t2)))
	# P1_possibilities.append(np.column_stack((R2,t1)))
	# P1_possibilities.append(np.column_stack((R2,t2)))

	# pclouds = []
	# for P1 in P1_possibilities:
	# 	pclouds.append(triangulate_points(im1_pts_ud_fixed, im2_pts_ud_fixed, P, P1))

	# infront_of_camera = []
	# for i in range(len(P1_possibilities)):
	# 	infront_of_camera.append(test_triangulation(P,pclouds[i])+test_triangulation(P1_possibilities[i],pclouds[i]))

	# best_pcloud = pclouds[np.argmax(infront_of_camera)]