# %%
# import the necessary packages
import numpy as np
import imutils
import cv2
from skimage.exposure import is_low_contrast

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import imutils
from imutils import perspective
from imutils import contours

# %%
# path = "live-view_fcc.mp4"
# vs = cv2.VideoCapture(path if path else 0)

frame = cv2.imread("frame536.jpg")

frame = imutils.resize(frame, width=450)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
edged = cv2.Canny(blurred, 30, 150)

# initialize the text and color to indicate that the current
# frame is *not* low contrast
text = "Low contrast: No"
color = (0, 255, 0)

# %%
# check to see if the frame is low contrast, and if so, update
# the text and color
if is_low_contrast(gray, fraction_threshold=0.35):
    text = "Low contrast: Yes"
    color = (0, 0, 255)
# otherwise, the frame is *not* low contrast, so we can continue
# processing it
else:
    # find contours in the edge map and find the largest one,
    # which we'll assume is the outline of our color correction
    # card
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # draw the largest contour on the frame
    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

# %%
# draw the text on the output frame
cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            color, 2)
# stack the output frame and edge map next to each other
output = np.dstack([edged] * 3)
# output = np.hstack([frame, output])

# %%
# show the output to our screen
# plt.imshow(imutils.opencv2matplotlib(output))
image = output
plt.imshow(imutils.opencv2matplotlib(image))


# %%

# # %%
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
# cv2.waitKey(0)
# cv2.imshow("Cropped ROI", roi)
# cv2.destroyAllWindows()


# y1=a
# y2=b
# x1=c
# x2=d
# # %%
# cv2.destroyAllWindows()

# %%
# #####################

y1=192 #a
y2=289 #b
x1=17 #c
x2=418 #d

#######################

# %%
roi = image[y1:y2, x1:x2]
# roi = image[192:289, 17:418]
roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


# %%
edgedc = edged[y1:y2, x1:x2]

# %%
image = roi.copy()
plt.imshow(imutils.opencv2matplotlib(image))

# # %%
# # for viewing all contours / debugging
# image2=image
# contours, hierarchy= cv2.findContours(edgedc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(image2, contours, -1, (0,255,0),1)
# plt.imshow(imutils.opencv2matplotlib(image2))



# %%

dilated = cv2.dilate(image, None, iterations=3) # 10 is an interesting number
eroded = cv2.erode(dilated, None, iterations=3)
plt.imshow(imutils.opencv2matplotlib(eroded))

# %%

# threshold the image by setting all pixel values less than 225 to 255 
# (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image 
# in short binary inversion
thresh = cv2.threshold(eroded, 225, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(imutils.opencv2matplotlib(thresh))

# %%
# we apply erosions to reduce the size of foreground objects
# further erosion to lose some unwanted small spaces
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=1)
plt.imshow(imutils.opencv2matplotlib(mask))
mask3 = mask.copy()

# %%
