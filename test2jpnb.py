# %%
#imports
# from skimage.filters import threshold_local
# import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours

# %%
pixelsPerMetric = 60
# %%

def midpoint(ptA, ptB): # to use in below function
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 
############################################

# %%
def croi2(cnts, oorig): # contour region of interest
	# loop over the contours individually
	for c in cnts:
		
		orig = oorig.copy()
		# if the contour is not sufficiently large, ignore it
		# print("Area of contour ", cv2.contourArea(c))
		if cv2.contourArea(c) < 100: # or cv2.contourArea(c) > 4000:
			continue
		
		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
	
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding box
	
		# box = perspective.order_points(box)
		# cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	#or
		box = perspective.order_points(box)
		box = box.astype("int")+[x1,y1]
		# asd
		# cv2.drawContours(orig, [box.astype("int")+[x1,y1]], -1, (0, 255, 0), 2)
	
		# roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
		roic = [
			[x1,y1],
			[x2,y1],
			[x2,y2],
			[x1,y2]
			]
		roic = np.array(roic, dtype="int")
		print(roic)
	
		cv2.drawContours(orig, [roic.astype("int")], -1, (0, 255, 0), 2)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	
	
		# print(box.astype("int"))
		# print(box.astype("int")+[x1, y1])
	
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	
		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
	
		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	
		# draw lines between the midpoints
		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)
	
		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	
		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / 10 # args["width"]
			print("pixelsPerMetric :",pixelsPerMetric)

	
		# compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric

	
		# draw the object sizes on the image
		cv2.putText(orig, "{:.2f}in".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)
		cv2.putText(orig, "{:.2f}in".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)
	
		# show the output image
		# plt.imshow(imutils.opencv2matplotlib(orig))
		cv2.imshow("Imager", orig)
		print("Area of contour ", cv2.contourArea(c))
	
		cv2.waitKey(0)
	cv2.destroyAllWindows()


############################################










# %%
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()


# ap.add_argument("-i", "--image", type=str, default="IMG_1097.JPG",
# 	help="path to input image")
# ap.add_argument("-w", "--width", type=float, default=10.0,
# 	help="width of the left-most object in the image (in inches)")

# args = vars(ap.parse_args())

# %%
%matplotlib inline

# load the input image
# image = cv2.imread(args["image"]) 
image = cv2.imread("IMG_1097.JPG")

image = imutils.resize(image, height = 500) # resize
oorig = image.copy()
plt.imshow(imutils.opencv2matplotlib(image))
# cv2.waitKey(0)

# %%
# # Select ROI dynamically in real time

# r = cv2.selectROI(image)
# cv2.waitKey(0)

# # Crop image
# roi = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# a = int(r[1])
# b = int(r[1]+r[3])
# c = int(r[0])
# d = int(r[0]+r[2])
# print(f"{a}:{b}, {c}:{d}")
# cv2.imshow("Cropped ROI", roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# y1=a
# y2=b
# x1=c
# x2=d

# %%
# #####################

y1=263 #a
y2=433 #b
x1=214 #c
x2=491 #d

#######################

# %%
roi = image[y1:y2, x1:x2]
# roi = image[207:479, 214:494]
# roi = image[263:433, 214:491]
roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


# %%
%matplotlib inline

image = roi.copy()
plt.imshow(imutils.opencv2matplotlib(image))

# %%

# def imgmod(image):
	# # %%

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(imutils.opencv2matplotlib(gray))
# cv2.imshow("Gray", gray)
# plt.imshow(imutils.opencv2matplotlib(image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # %%

# apply a Gaussian blur with a 7x7 kernel to the image to smooth it,
# reducing high frequency noise
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
plt.imshow(imutils.opencv2matplotlib(blurred))



# # %%

# applying edge detection
# edged = cv2.Canny(gray, 200, 255) 
# edged = cv2.Canny(blurred, 100, 200) # blurred
edged = cv2.Canny(blurred, 50, 100) # blurred
plt.imshow(imutils.opencv2matplotlib(edged))


# # %%
# # Autocanny 
# # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
# edgeMap = imutils.auto_canny(blurred)
# # cv2.imshow("Original", image)
# plt.imshow(imutils.opencv2matplotlib(edgeMap))


# # %%

dilated = cv2.dilate(edged, None, iterations=2) # 10 is an interesting number
eroded = cv2.erode(dilated, None, iterations=2)
plt.imshow(imutils.opencv2matplotlib(eroded))


# # %%

# threshold the image by setting all pixel values less than 225 to 255 
# (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image 
# in short binary inversion
thresh = cv2.threshold(eroded, 225, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(imutils.opencv2matplotlib(thresh))

# # %%
# we apply erosions to reduce the size of foreground objects
# further erosion to lose some unwanted small spaces
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=1)
plt.imshow(imutils.opencv2matplotlib(mask))


# # %%

# # for viewing all contours / debugging
# image2=image
# contours, hierarchy= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(image2, contours, -1, (0,255,0),1)
# plt.imshow(imutils.opencv2matplotlib(image2))

	# return mask

# %%
# mask = imgmod(image)


# %%
# find contours in the edge map
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

# TODO: needs to be fine tuned
# pixelsPerMetric = 60 # hard coded value

# %%
# %matplotlib widget
tc = []
# loop over the contours individually
for c in cnts:

	orig = oorig.copy()
	
	# roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
	roic = [
        [x1,y1],
        [x2,y1],
        [x2,y2],
     	[x1,y2]
        ]
	roic = np.array(roic, dtype="int")
	# print(roic)

	# if the contour is not sufficiently large, ignore it
	# print("Area of contour ", cv2.contourArea(c))
	if cv2.contourArea(c) < 3000 or cv2.contourArea(c) > 5000:
		continue


	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
 
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding box
 
	# box = perspective.order_points(box)
	# cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
 #or
	box = perspective.order_points(box)
	box = box.astype("int")+[x1,y1]
	# asd
	# cv2.drawContours(orig, [box.astype("int")+[x1,y1]], -1, (0, 255, 0), 2)
 
 
	cv2.drawContours(orig, [roic.astype("int")], -1, (255, 255, 0), 2)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
 
###########################################################
	# print(box.astype("int"))
	# print(box.astype("int")+[x1, y1])
 
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
  
	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	# print(box)
	print("tl", tl)
	print("tr", tr)
	print("br", br)
	print("bl", bl)

	cv2.imshow("Imager", orig)
	print("Area of contour ", cv2.contourArea(c))
 
	tc.append(box)
	cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
tc # tile contours with rectangle border
# %%

for t in tc:
	# print(c)
	image_tile = oorig
	roi_tile = image_tile[t[0][1]:t[2][1], t[0][0]:t[2][0]]

	cv2.imshow("roi_tile", roi_tile)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	gray = cv2.cvtColor(roi_tile, cv2.COLOR_BGR2GRAY)
	# plt.imshow(imutils.opencv2matplotlib(gray))
	# cv2.imshow("Gray", gray)
	# plt.imshow(imutils.opencv2matplotlib(image))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# # %%

	# apply a Gaussian blur with a 7x7 kernel to the image to smooth it,
	# reducing high frequency noise
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	# plt.imshow(imutils.opencv2matplotlib(blurred))



	# # %%

	# applying edge detection
	# edged = cv2.Canny(gray, 200, 255) 
	# edged = cv2.Canny(blurred, 100, 200) # blurred
	edged = cv2.Canny(blurred, 50, 100) # blurred
	# plt.imshow(imutils.opencv2matplotlib(edged))


	# # %%
	# # Autocanny 
	# # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	# edgeMap = imutils.auto_canny(blurred)
	# # cv2.imshow("Original", image)
	# plt.imshow(imutils.opencv2matplotlib(edgeMap))


	# # %%

	dilated = cv2.dilate(edged, None, iterations=2) # 10 is an interesting number
	eroded = cv2.erode(dilated, None, iterations=2)
	# plt.imshow(imutils.opencv2matplotlib(eroded))


	# # %%

	# threshold the image by setting all pixel values less than 225 to 255 
	# (white; foreground) and all pixel values >= 225 to 255
	# (black; background), thereby segmenting the image 
	# in short binary inversion
	thresh = cv2.threshold(eroded, 225, 255, cv2.THRESH_BINARY_INV)[1]
	# plt.imshow(imutils.opencv2matplotlib(thresh))

	# # %%
	# we apply erosions to reduce the size of foreground objects
	# further erosion to lose some unwanted small spaces
	mask = thresh.copy()
	mask = cv2.erode(mask, None, iterations=1)
	# plt.imshow(imutils.opencv2matplotlib(mask))


	# # %%
	# find contours in the edge map
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)


	for c in cnts:
		
		orig = oorig.copy()
		# if the contour is not sufficiently large, ignore it
		# print("Area of contour ", cv2.contourArea(c))
		if cv2.contourArea(c) < 100: # or cv2.contourArea(c) > 4000:
			continue
		
		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		print(box)
	
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding box
	
		# box = perspective.order_points(box)
		# cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	#or
		box = perspective.order_points(box)
		box = box.astype("int")+[t[0][0],t[0][1]] #[x1,y1]
		# asd
		# cv2.drawContours(orig, [box.astype("int")+[x1,y1]], -1, (0, 255, 0), 2)
	
		# roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
		# roic = [
		# 	[x1,y1],
		# 	[x2,y1],
		# 	[x2,y2],
		# 	[x1,y2]
		# 	]
		# roic = np.array(roic, dtype="int")
		# print(roic)
	
		cv2.drawContours(orig, [roic.astype("int")], -1, (0, 255, 0), 2)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	
	
		# print(box.astype("int"))
		# print(box.astype("int")+[x1, y1])
	
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	
		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
	
		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	
		# draw lines between the midpoints
		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)
	
		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	
		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / 10 # args["width"]
			print("pixelsPerMetric :",pixelsPerMetric)

	
		# compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric

	
		# draw the object sizes on the image
		cv2.putText(orig, "{:.2f}in".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)
		cv2.putText(orig, "{:.2f}in".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)
	
		# show the output image
		# plt.imshow(imutils.opencv2matplotlib(orig))
		cv2.imshow("Imager", orig)
		print("Area of contour ", cv2.contourArea(c))
	
		cv2.waitKey(0)
	cv2.destroyAllWindows()


# cv2.destroyAllWindows()
# %%