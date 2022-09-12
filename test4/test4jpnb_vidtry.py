# %%
# import the necessary packages
import numpy as np
import imutils
import cv2
from skimage.exposure import is_low_contrast
import argparse
import time

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import imutils
from imutils import perspective
from imutils import contours

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="optional path to video file")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
                help="threshold for low contrast")
args = vars(ap.parse_args())

# # %%
print("[INFO] accessing video stream...")
path = "live-view_fcc.mp4"
# vs = cv2.VideoCapture(path if path else 0)
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
start_frame_number = 100
# vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

while True:
    # frame = cv2.imread("frame536.jpg")
    cv2.waitKey(100)
    (grabbed, frame) = vs.read()

    if not grabbed:
        print("[INFO] no frame read from stream - exiting")
        break
    
    frame = imutils.resize(frame, width=450)
    # plt.imshow(imutils.opencv2matplotlib(frame))

    # # %%
    image = frame.copy()
    oorig = frame.copy()

    # # %%
    # roi = frame.copy()
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

    # # %%
    # # #####################

    # # %%
    y1=194 #a
    y2=287 #b
    x1=25 #c
    x2=416 #d

    # #######################

    # # %%
    # roi = image[y1:y2, x1:x2]
    # # roi = image[192:289, 17:418]
    # # roi = image[194:287, 25:416]
    # roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    # image=roi.copy()
    # plt.imshow(imutils.opencv2matplotlib(image))



    # # %%
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # plt.imshow(imutils.opencv2matplotlib(gray))

    # # %%
    blurred = cv2.GaussianBlur(gray.copy(), (11, 11), 0)
    # plt.imshow(imutils.opencv2matplotlib(blurred))


    # # %%
    edged = cv2.Canny(blurred.copy(), 30, 150)
    # plt.imshow(imutils.opencv2matplotlib(edged))

    # # %%
    # initialize the text and color to indicate that the current
    # frame is *not* low contrast
    # text = "Low contrast: No"
    # color = (0, 255, 0)

    # check to see if the frame is low contrast, and if so, update
    # the text and color
    if is_low_contrast(gray, fraction_threshold=0.35):
        # text = "Low contrast: Yes"
        # color = (0, 0, 255)
        pass
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

    # # draw the text on the output frame
    # cv2.putText(frame, 
    #             text, 
    #             (5, 25), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 
    #             0.8,
    #             color, 
    #             2)
    # stack the output frame and edge map next to each other
    edged3 = np.dstack([edged] * 3)
    # output = np.hstack([frame, output])

    # show the output to our screen
    # plt.imshow(imutils.opencv2matplotlib(output))
    # image = output.copy()
    # plt.imshow(imutils.opencv2matplotlib(edged3))

    # # %%

    dilated = cv2.dilate(edged.copy(), None, iterations=3) # 10 is an interesting number
    eroded = cv2.erode(dilated, None, iterations=3)
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
    mask3 = mask.copy()


    # # %%
    # find contours in the edge map
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)

    # TODO: needs to be fine tuned
    # pixelsPerMetric = 60 # hard coded value


    # # %%
    # # roi = image[y1:y2, x1:x2].copy()
    # edgedc = edged3[y1:y2, x1:x2].copy()
    # # roi = image[192:289, 17:418]
    # roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    # plt.imshow(imutils.opencv2matplotlib(edgedc))


    # # %%
    # # for viewing all contours / debugging
    # image2=image.copy()
    # contours, hierarchy= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image2, contours, -1, (0,255,0),1)
    # plt.imshow(imutils.opencv2matplotlib(image2))


    # # %%
    # # %%
    # # contour filtering and approximation
    orig = oorig.copy()
    tc = []
    tca = []
    abcde = []
    pixelsPerMetric = 110

    for c in cnts:
        # print("contour area=",cv2.contourArea(c))
        if cv2.contourArea(c) < 13000 or cv2.contourArea(c) > 14000:
            continue

        output = oorig.copy()
        cv2.drawContours(output, [c], -1, (0, 255, 0), 3)
        (x, y, w, h) = cv2.boundingRect(c)
        text = "original, num_pts={}".format(len(c))
        cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 0), 2)

        # show the original contour image
        # print("[INFO] {}".format(text))
        # cv2.imshow("Original Contour", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(imutils.opencv2matplotlib(output))


        # for eps in np.linspace(0.001, 0.05, 10):
        #     # approximate the contour
        #     # eps=0.0228
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, eps * peri, True)
        #     # abcde.append(approx)

        #     # # draw the approximated contour on the image
        #     output = image.copy()
        #     cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
        #     text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
        #     cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
        #     	0.4, (0, 255, 0), 2)
        #     # show the approximated contour image
        #     print("[INFO] {}".format(text))
        #     cv2.imshow("Approximated Contour", output)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        eps=0.0228
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        abcde.append(approx)

        # mod abcde
        orig = oorig.copy()

        rabcde = abcde #bkp
        for approxa in abcde:
            for ppopints in range((len(approxa)//2)):
                # print(ppopints)

                # tremp=(-ppopints)-1
                # print(tremp)

                # approxa[-ppopints][0][0]=approxa[ppopints][0][0] #old

                # comment this if-else statement to see the actual situation
                if ppopints>2:
                    approxa[(-ppopints)-1][0][0]=approxa[ppopints][0][0]
                else:
                    approxa[ppopints][0][0]=approxa[(-ppopints)-1][0][0]

                cv2.circle(orig, approxa[ppopints][0], 3, (255, 0, 0), -1)
                cv2.circle(orig, approxa[(-ppopints)-1][0], 3, (0, 0, 255), -1)
            # cv2.imshow("points", orig)
            # cv2.waitKey(0)
            # plt.imshow(imutils.opencv2matplotlib(orig))
        # cv2.destroyAllWindows()

        # abcde
        orig = oorig.copy()

        approxac=0
        for approxa in abcde:
            for popints in range(len(approxa)//2):
                # print(approxa[popints], approxa[-popints])
                # print(popints)
                # print("len(approxa//2:",len(approxa)//2)
                # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                # dA = dist.euclidean(rabcde[approxac][popints][0], rabcde[approxac][(-popints)-1][0])
                dA = dist.euclidean(abcde[approxac][popints][0], abcde[approxac][(-popints)-1][0]) # or just use this to measure om modified co-ords
                dimA = dA / pixelsPerMetric

                cv2.line(orig, approxa[popints][0], approxa[(-popints-1)][0], (0, 255, 0), 1)
                cv2.circle(orig, approxa[popints][0], 3, (0, 255, 0), -1)
                cv2.circle(orig, approxa[(-popints)-1][0], 3, (0, 255, 0), -1)
                # if popints!=1:
                    # continue
                cv2.putText(orig, "{:.2f}mm".format((dimA*2.54)*10+4),
                            # (int(tl[0])-15, int(tl[1])-15), 
                            approxa[popints][0]-[25,15], 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 255, 0), 1)

                # stack the output frame and edge map next to each other
                # output = np.dstack([edged] * 3)
                # output = np.hstack([frame, orig])
                output = np.hstack(orig)
                # show the output to our screen
                cv2.imshow("Output", output)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            approxac+=1
        # cv2.imshow("measurements", orig)
        # cv2.waitKey(0)
        # plt.imshow(imutils.opencv2matplotlib(orig))
        # cv2.destroyAllWindows()

# # %%

# %%
