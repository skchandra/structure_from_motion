import cv2
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#camera matrix and distortion values to use to undistort images 
D = np.array( [0.07449795106852891, -0.26800080727246195, 0.00010699178813798149, -0.0016215045904716067, 0.0])
K = np.array( [[649.0759367719918, 0.0, 313.97037517958785], [0.0, 645.5178014943875, 230.33935085177768], [0.0, 0.0, 1.0]])
W = np.array([[0.0, -1.0, 0.0],
			  [1.0, 0.0, 0.0],
			  [0.0, 0.0, 1.0]])

def draw_flow(im,flow,step=16):
	"""Function to visualize optical flow on the 2nd image in order to see change. 
	Input: 2nd image, optical flow array, step
	Output: 2nd image with optical flow lines.
	source: http://stackoverflow.com/questions/13685771/opencv-python-calcopticalflowfarneback"""

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
	"""Calculates most likely projection matrix of the objects. 
	Input: List of possible projection matrices, point cloud
	Output: The average of all the z coordinates in the point cloud (if they are greater than 0)
	source: show_depth.py (included in Git repo) from Paul Ruvolo"""

	P4x4 = np.eye(4)	
	P4x4[0:3,:] = P
	pcloud_3d = pcloud[:,0:3]
	projected = cv2.perspectiveTransform(np.array([pcloud_3d]),P4x4)
	return np.mean(projected[0,:,2]>0.0)

def construct(im1, im2):
	"""Function to calculate optical flow between two images and extrapolate x, y, and z coordinates of each pixel
	that has moved above a certain threshold. Also produces a 3d plot of the object(s).
	Input: 2 images of the same objects
	Output: none """

	old_image = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	new_image = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	#calculates flow and calls function to draw the flow vectors on the image
	flow = cv2.calcOpticalFlowFarneback(old_image, new_image, 0.5, 3, 12, 3, 5, 1.2, 0)
	cv2.imshow('Optical flow',draw_flow(new_image,flow))

	correspondences = ([], [])	#array of matched keypoints between two images

	#iterate through first image and get optical flow for every pixel in every 3rd row and 3rd column, if it is larger than threshold
	for y in range(0, old_image.shape[0], 3):
			for x in range(0, old_image.shape[1], 3):
				if abs(flow[y][x][0]) > 0.3 and abs(flow[y][x][1]) > 0.3:
					correspondences[0].append((y,x))
					correspondences[1].append((y + flow[y][x][0],
													x + flow[y][x][1]))

	#source of following code: show_depth.py from Paul Ruvolo
	im1_pts = np.zeros((len(correspondences[0]),2))
	im2_pts = np.zeros((len(correspondences[1]),2))

	#creating a visualization of the keypoints on the first image
	im = np.array(np.hstack((im1,im2)))
	for i in range(len(correspondences[0])):
		cv2.circle(im, (int(correspondences[0][i][1]), int(correspondences[0][i][0])),1,(0,255,255),1)
	cv2.imshow("keypoints",im)

	# plot the points
	for i in range(len(im1_pts)):
		im1_pts[i,0] = correspondences[0][i][0]
		im1_pts[i,1] = correspondences[0][i][1]
		im2_pts[i,0] = correspondences[1][i][0]
		im2_pts[i,1] = correspondences[1][i][1]


	im1_pts_augmented = np.zeros((1,im1_pts.shape[0],im1_pts.shape[1]))
	im1_pts_augmented[0,:,:] = im1_pts
	im2_pts_augmented = np.zeros((1,im2_pts.shape[0],im2_pts.shape[1]))
	im2_pts_augmented[0,:,:] = im2_pts

	im1_pts_ud = cv2.undistortPoints(im1_pts_augmented,K,D)
	im2_pts_ud = cv2.undistortPoints(im2_pts_augmented,K,D)
	#end sourced code
	
	#calculate essential matrix
	F, mask = cv2.findFundamentalMat(im1_pts_ud, im2_pts_ud, cv2.FM_RANSAC, 3, 0.99)
	E = K.T * F * K
	W_inv = np.linalg.inv(W)

	# get rotational/translational components from E
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

	P1 = np.eye(3, 4, dtype=np.float64)   #identity projection matrix

	#all possibilities for P2
	P2 = np.array([[R1[0][0], R1[0][1], R1[0][2], T1[0]],
					[R1[1][0], R1[1][1], R1[1][2], T1[1]],
					[R1[2][0], R1[2][1], R1[2][2], T1[2]]])

	P2_possibilities = []
	P2_possibilities.append(np.column_stack((R1,T1)))
	P2_possibilities.append(np.column_stack((R1,T2)))
	P2_possibilities.append(np.column_stack((R2,T1)))
	P2_possibilities.append(np.column_stack((R2,T2)))

	#triangulate points for every possible projection matrix
	cloud = []
	for i in P2_possibilities:
		cloud.append(cv2.triangulatePoints(P1, i, im1_pts_ud, im2_pts_ud))

	#finding most likely projection matrix for each possibility. 
	#source: show_depth.py from Paul Ruvolo
	infront_of_camera = []
	for i in range(len(P2_possibilities)):
		infront_of_camera.append(test_triangulation(P1, cloud[i]) + test_triangulation(P2_possibilities[i], cloud[i]))

	cloud = cloud[np.argmax(infront_of_camera)]
	#end sourced code

	output = np.array(cloud)
	output = output.T 	#transpose to change format of output
	x = []
	y = []
	z = []

	#normalizing the x, y, z coordinates
	for i in output:
		print i
		i[0] = i[0]/i[3]
		x.append(i[0])
		i[1] = i[1]/i[3]
		y.append(i[1])
		i[2] = i[2]/i[3]
		z.append(i[2])
		i[3] = i[3]/i[3]
	
	#plotting the object in a scatter plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z, c='b')
	plt.show()


# old function with some previous calculations that we tried


	#	def optical_flow(self):
		# sift = cv2.SIFT()

		# old_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
		# new_image = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

		# pts0, des0 = sift.detectAndCompute(old_image, None)
		# pts1, des1 = sift.detectAndCompute(new_image, None)

		# bf = cv2.BFMatcher()
		# matches = bf.knnMatch(des0, des1, k=2)

		# good = []
		# for m,n in matches:
		#     if m.distance < 50:
		#         good.append([m])

		# # Apply ratio test
		# for m,n in matches:
		#     if m.distance < 0.75*n.distance:
		#     	self.correspondences[0].append((m.queryIdx, m.trainIdx))
		#     	self.correspondences[1].append((n.queryIdx, n.trainIdx))

		# old_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
		# new_image = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
		# # # cv2.imshow('distorted', old_image)
		# # # old_image = cv2.undistort(old_image,K,D)
		# # # cv2.imshow('undistort', old_image)
		# # # new_image = cv2.undistort(new_image,K,D)
		# flow = cv2.calcOpticalFlowFarneback(old_image, new_image, 0.5, 3, 12, 3, 5, 1.2, 0)
		
		# # create array of points
		# for y in range(0, old_image.shape[0], 6):
		# 	for x in range(0, old_image.shape[1], 6):
		# 		if abs(flow[y][x][0] > 0.1) and abs(flow[y][x][1] > 0.1):
		# 			self.correspondences[0].append((y,x))
		# 			self.correspondences[1].append((y + flow[y][x][0],
		# 											x + flow[y][x][1]))
		#construct(self.image1, self.image2)
		
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

		# Keypoint matching
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