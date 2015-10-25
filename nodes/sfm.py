import cv2
import numpy as np 

D = np.array( [0.12969982463039115, -0.3662323365747535, -0.0047588179431673535, 0.0032280633422334627, 0.0])
K = np.array( [[672.9140718714626, 0.0, 317.4277026418419], [0.0, 670.9812424489073, 216.90441559892045], [0.0, 0.0, 1.0]])
W = np.array([[0.0, -1.0, 0.0],
			  [1.0, 0.0, 0.0],
			  [0.0, 0.0, 1.0]])

def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis

def test_triangulation(P,pcloud):
	P4x4 = np.eye(4)
	P4x4[0:3,:] = P
	pcloud_3d = pcloud[:,0:3]
	projected = cv2.perspectiveTransform(np.array([pcloud_3d]),P4x4)
	return np.mean(projected[0,:,2]>0.0)

def construct(im1, im2):
	old_image = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	new_image = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(old_image, new_image, 0.5, 3, 12, 3, 5, 1.2, 0)
	cv2.imshow('Optical flow',draw_flow(new_image,flow))
	correspondences = ([], [])

	for y in range(0, old_image.shape[0], 1):
			for x in range(0, old_image.shape[1], 1):
				if abs(flow[y][x][0]) > 0.3 and abs(flow[y][x][1]) > 0.3:
					correspondences[0].append((y,x))
					correspondences[1].append((y + flow[y][x][0],
													x + flow[y][x][1]))

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

	
	F, mask = cv2.findFundamentalMat(im1_pts_ud, im2_pts_ud, cv2.FM_RANSAC, 3, 0.99)
	E = K.T * F * K
	W_inv = np.linalg.inv(W)

	U, S, V = np.linalg.svd(E)

	R1 = U * W * V.T
	if np.linalg.det(R1)+1 < 10**-8:
		F = -F
		E = K.T * F * K
		U, S, V = np.linalg.svd(E)
		R1 = U * W * V.T
	T1 = [i[2] for i in U]

	R2 = U * W_inv * V.T
	T2 = [-i[2] for i in U]

	P1 = np.eye(3, 4, dtype=np.float64)
	P2 = np.array([[R1[0][0], R1[0][1], R1[0][2], T1[0]],
					[R1[1][0], R1[1][1], R1[1][2], T1[1]],
					[R1[2][0], R1[2][1], R1[2][2], T1[2]]])

	P2_possibilities = []
	P2_possibilities.append(np.column_stack((R1,T1)))
	P2_possibilities.append(np.column_stack((R1,T2)))
	P2_possibilities.append(np.column_stack((R2,T1)))
	P2_possibilities.append(np.column_stack((R2,T2)))

	cloud = []
	for i in P2_possibilities:
		cloud.append(cv2.triangulatePoints(P1, i, im1_pts_ud, im2_pts_ud))

	infront_of_camera = []
	for i in range(len(P2_possibilities)):
		infront_of_camera.append(test_triangulation(P1, cloud[i]) + test_triangulation(P2_possibilities[i], cloud[i]))

	cloud = cloud[np.argmax(infront_of_camera)]

	output = np.array(cloud)
	output = output.T
	x = []
	y = []
	z = []
	for i in output:
		x.append(i[0])
		y.append(i[1])
		z.append(i[2])

	for i in range(len(correspondences[0])):
		cv2.circle(im, (int(correspondences[0][i][1]), int(correspondences[0][i][0])),1,(0,255,0),1)

	cv2.imshow("MYWIN",im)
	# for i in output:
	# 	print i
	# 	i[0] = i[0]/i[3]
	# 	x.append(i[0])
	# 	i[1] = i[1]/i[3]
	# 	y.append(i[1])
	# 	i[2] = i[2]/i[3]
	# 	z.append(i[2])
	# 	i[3] = i[3]/i[3]

	# print output
	
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z)
	plt.show()

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