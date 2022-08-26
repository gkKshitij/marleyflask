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
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
output = np.hstack([frame, output])

# %%
# show the output to our screen
plt.imshow(imutils.opencv2matplotlib(output))

# %%
