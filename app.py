# from flask import Flask
from itertools import count
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,flash 
from werkzeug.utils import secure_filename
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import statistics

# from skimage.exposure import is_low_contrast

# arrangements for heroku
UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
PROCESSED = 'static/downloads/processed/'
ALLOWED_EXTENSIONS = {'jpg', 'png','jpeg', 'JPG', 'PNG'}
app = Flask(__name__, static_url_path="/static")


# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'opencv'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 6mb
app.config['MAX_CONTENT_LENGTH'] = 7 * 1024 * 1024

counter = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(path, filename):
    if filename!="frame536.jpg":
        detect_object(path, filename)
    else:
        cabcde=detect_object2(path, filename)
        return cabcde
    
###
def midpoint(ptA, ptB): # to use in below function
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

filelist = []

def detect_object2(path, filename):
    frame = cv2.imread(path)
    frame = imutils.resize(frame, width=450)

    image = frame.copy()
    oorig = frame.copy()

    
    # # %%
    # y1=194 #a
    # y2=287 #b
    # x1=25 #c
    # x2=416 #d

    # #######################

    # # %%
    # roi = image[y1:y2, x1:x2]
    # # roi = image[192:289, 17:418]
    # # roi = image[194:287, 25:416]
    # roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


    # %%
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray.copy(), (11, 11), 0)
    edged = cv2.Canny(blurred.copy(), 30, 150)

    dilated = cv2.dilate(edged.copy(), None, iterations=3) # 10 is an interesting number
    eroded = cv2.erode(dilated, None, iterations=3)

    thresh = cv2.threshold(eroded, 225, 255, cv2.THRESH_BINARY_INV)[1]

    mask = thresh.copy()
    mask = cv2.erode(mask, None, iterations=1)
    mask3 = mask.copy()

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)

    # TODO: needs to be fine tuned
    pixelsPerMetric = 110 # hard coded value

    # # contour filtering and approximation
    orig = oorig.copy()
    tc = []
    tca = []
    abcde = []
    cabcde = [] # calculated abcde

    aroe = [ # allowed range of error
    [10.5,11],
    [12.0,13.0],
    [11.0,12.0],
    [12.0,13.0],
    [11.0,12.0],
    ]

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

        eps=0.0228
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        abcde.append(approx)

        # mod abcde
        orig = oorig.copy()

        rabcde = abcde #bkp
        for approxa in abcde:
            for ppopints in range((len(approxa)//2)):
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

        pnt=0 # point

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
     
                red = (0,0,255)
                green = (0,255,0)

                dim = ((dimA*2.54)*10+4)
                

                # print(aroe[4-pnt][0], dim, aroe[4-pnt][1])
                # print(dim>aroe[4-pnt][0], dim, dim<aroe[4-pnt][1])
                # print("\n")
                tc = red
                if (dim>aroe[4-pnt][0]) and (dim<aroe[4-pnt][1]):
                    tc = green

                cv2.putText(orig, "{:.2f}mm".format(dim),
                            # (int(tl[0])-15, int(tl[1])-15), 
                            approxa[popints][0]-[25,15], 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, tc, 1)
                pnt+=1
                cabcde.append(dim)      
            approxac+=1

        # cv2.imshow("measurements", orig)
        # cv2.waitKey(0)
        # plt.imshow(imutils.opencv2matplotlib(orig))
        # cv2.destroyAllWindows()
        greendots = orig.copy()
        cv2.imwrite(f"{DOWNLOAD_FOLDER}processed/greendotsi_{filename}",orig)
    return(cabcde)




def detect_object(path, filename):    # TODO:
    
    image = cv2.imread(path)
    # print(path)
    # print(filename)

    
    # image = cv2.resize(image,(480,360))
    # (h, w) = image.shape[:2]
    image = imutils.resize(image, height = 500) # resize
    oorig = image.copy()

    
################################
##
##

    # y1=327 #a
    # y2=385 #b
    # x1=218 #c
    # x2=491 #d

    y1=263 #a
    y2=433 #b
    x1=214 #c
    x2=491 #d

    roi = image[y1:y2, x1:x2]
    image = roi.copy()
    roic = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


    # DOWNLOAD_FOLDER = 'static/downloads/'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{DOWNLOAD_FOLDER}gray_{filename}",gray)
    
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imwrite(f"{DOWNLOAD_FOLDER}blurred_{filename}",blurred)
    edged = cv2.Canny(blurred, 50, 150)
    eroi = edged.copy()
    cv2.imwrite(f"{DOWNLOAD_FOLDER}edged_{filename}",edged)
    dilated = cv2.dilate(edged, None, iterations=4) # 10 is an interesting number
    cv2.imwrite(f"{DOWNLOAD_FOLDER}dilated_{filename}",dilated)
    eroded = cv2.erode(dilated, None, iterations=4)
    cv2.imwrite(f"{DOWNLOAD_FOLDER}eroded_{filename}",eroded)
    thresh = cv2.threshold(eroded, 225, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite(f"{DOWNLOAD_FOLDER}thresh_{filename}",thresh)

    mask = thresh.copy()
    mask = cv2.erode(mask, None, iterations=1)
    cv2.imwrite(f"{DOWNLOAD_FOLDER}mask_{filename}",mask)
    mask3 = mask.copy()


    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = 60 # hard coded value

    orig = oorig.copy()
    tc = []
    tca = []
    abcde = []

    # loop over the contours individually
    nooftiles=0
    for c in cnts:

        # orig = oorig.copy()
        
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
        if cv2.contourArea(c) < 4000  or cv2.contourArea(c) > 5200:
            continue
        # print("Area of contour ", cv2.contourArea(c))

        tca.append(c)
        
        eps=0.0228
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        abcde.append(approx)



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
        
        nooftiles+=1

        # print(box)
        # print("tl", tl)
        # print("tr", tr)
        # print("br", br)
        # print("bl", bl)

        # print(nooftiles)
        nooftiles+=1
        
        cv2.putText(orig, f"{nooftiles}",
                    (int(tl[0])+15, int(tl[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2)

        # cv2.imshow("Imager", orig)
        # print("Area of contour ", cv2.contourArea(c))
        # print(tc)
        tc.append(box)
        cv2.waitKey(0)
        tcai = orig.copy()


        if cv2.contourArea(c) < 5200: # temp jugad cause code is rendered differently on server
            tc.append(box)
            #continue
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"{DOWNLOAD_FOLDER}tilesdetected_{filename}",orig)


    counter = 0

    for t in tc:
        # print(c)
        # t=1
        image_tile = oorig
        roi_tile = image_tile[t[0][1]:t[2][1], t[0][0]:t[2][0]]

        # cv2.imshow("roi_tile", roi_tile)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

        orig2 = oorig.copy()
        orig = oorig.copy()
        countertemp = 0


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
            # print(box)
        
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
        
            cv2.drawContours(orig, [roic.astype("int")], -1, (255, 255, 0), 2)
            cv2.drawContours(orig2, [roic.astype("int")], -1, (255, 255, 0), 2)
            # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            
            countertemp += 1

        
        
            # print(box.astype("int"))
            # print(box.astype("int")+[x1, y1])
        
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.circle(orig2, (int(x), int(y)), 3, (0, 0, 255), -1)
        
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

            cv2.line(orig, tl, bl, (0, 0, 255), 1) 
            cv2.line(orig, tr, br, (0, 0, 255), 1) 

            cv2.line(orig2, tl, bl, (0, 0, 255), 1) 
            cv2.line(orig2, tr, br, (0, 0, 255), 1) 
        
            # # draw the midpoints on the image
            # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
            # # draw lines between the midpoints
            # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            #     (255, 0, 255), 2)
            # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            #     (255, 0, 255), 2)
        
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                pixelsPerMetric = 60 #dB / 10 # args["width"]
                # print("pixelsPerMetric :",pixelsPerMetric)

        
            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

        
            # draw the object sizes on the image
            # draw the object sizes on the image
            cv2.putText(orig, "{:.2f}mm".format((dimA*2.54)*10),
                (int(tl[0])-15, int(tl[1])-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2)
            cv2.putText(orig, "{:.2f}mm".format((dimA*2.54)*10),
                (int(tr[0])-15, int(tr[1])-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2)
            cv2.putText(orig2, "{:.2f}mm".format((dimA*2.54)*10),
                (int(tl[0])-15, int(tl[1])-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2)
            cv2.putText(orig2, "{:.2f}mm".format((dimA*2.54)*10),
                (int(tr[0])-15, int(tr[1])-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2)
                
            # cv2.putText(orig, "{:.2f}in".format(dimA),
            #     (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.35, (0, 0, 255), 1)
            # cv2.putText(orig, "{:.2f}in".format(dimB),
            #     (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.35, (0, 0, 255), 1)
        
            # show the output image
            # plt.imshow(imutils.opencv2matplotlib(orig))
            # cv2.imshow("Imager", orig)
            # print("Area of contour ", cv2.contourArea(c))
            counter += 1
            # print(counter)
            # cv2.waitKey(0)
            cv2.imwrite(f"{DOWNLOAD_FOLDER}processed/c{counter}_{filename}",orig)
            cv2.imwrite(f"{DOWNLOAD_FOLDER}processed/caio_{filename}",orig2)
            contournp = (f"{DOWNLOAD_FOLDER}processed/c{counter}_{filename}")
            # print(contournp)
            filelist.append(contournp)
            # print(counter)

        # mod abcde
        orig = oorig.copy()

        rabcde = abcde
        for approxa in abcde:
            for ppopints in range(1,len(approxa),2):
                approxa[-ppopints][0][0]=approxa[ppopints][0][0]


    # abcde
        approxac=0
        for approxa in abcde:
            for popints in range(1,len(approxa)//2):
                # print(approxa[popints], approxa[-popints])
                # print(popints)
                # print("len(approxa//2:",len(approxa)//2)
                # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dA = dist.euclidean(rabcde[approxac][popints][0], rabcde[approxac][-popints][0])
                dimA = dA / pixelsPerMetric

                cv2.line(orig, approxa[popints][0]+[x1,y1], approxa[-popints][0]+[x1,y1], (0, 255, 0), 1)
                cv2.circle(orig, approxa[popints][0]+[x1,y1], 3, (0, 255, 0), -1)
                cv2.circle(orig, approxa[-popints][0]+[x1,y1], 3, (0, 255, 0), -1)
                # if popints!=1:
                    # continue

                cv2.putText(orig, "{:.2f}mm".format((dimA*2.54)*10+4),
                            # (int(tl[0])-15, int(tl[1])-15), 
                            approxa[popints][0]+[x1,y1], 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, (0,255,0), 0), 1)
                # pnt+=1
                
            approxac+=1
        greendots = orig.copy()
        cv2.imwrite(f"{DOWNLOAD_FOLDER}processed/greendotsi_{filename}",orig)



        # cv2.destroyAllWindows()
    
    # def it here
    #2b starts here segmentation to median
    # for i in range((len(abcde)))
    bu=[]
    bd=[]
    cu=[]
    cd=[]
    du=[]
    dd=[]

    for approxa in abcde:
        # for popints in range(1,len(approxa)//2):
        bu.append(approxa[1][0])
        bd.append(approxa[-1][0])
        cu.append(approxa[2][0])
        cd.append(approxa[-2][0])
        du.append(approxa[3][0])
        dd.append(approxa[-3][0])

    orig=oorig.copy()

    orig=oorig.copy()
    thickness=2
    n=10
    # pu=du
    # pd=dd
    pu=[bu, cu, du]
    pd=[bu, cd, dd]
    # x1=x1,y1=y1,x2=x2,y2=y2, n=10,thickness, oorig=oorig

###############################
    # def bcd_lr(u, d,thickness,x1=x1,y1=y1,x2=x2,y2=y2, n=10, oorig=oorig):
    #     # # %%
    #     t=thickness
    #     pu = du
    #     pd = dd

    #     coordsl = []
    #     coordsr = []

    #     dimSr = []
    #     dimSl = []

    #     # # %%
    #     for i in range(n):
    #         # dA = dist.euclidean(rabcde[approxac][popints][0], rabcde[approxac][-popints][0])
    #         # dimA = dA / pixelsPerMetric
    #         # coordsl.append([pu[0]+[-(i*t),15], pd[0]+[-(i*t),-25]]) # l				
    #         # coordsr.append([pu[0]+[(i*t),15], pd[0]+[(i*t),-25]]) # r
    #         coordsl.append([pu[0]+[x1-(i*t),y1+15], pd[0]+[x1-(i*t),y1-25]]) # l				
    #         coordsr.append([pu[0]+[x1+(i*t),y1+15], pd[0]+[x1+(i*t),y1-25]]) # r

    #     # #  %%

    #     # cv2.line(maskt, coordsr[0][0], coordsr[0][1], (0, 0, 255), 1)
    #     # cv2.line(maskt, coordsr[0][0]+[1,0], coordsr[0][1]+[1,0], (0, 0, 255), 1)
    #     # plt.imshow(imutils.opencv2matplotlib(origs))


    #     # # %%
    #     for j in range(n-1):
    #         # print("j=",j)
    #         origs = oorig.copy()
    #         y1s=coordsr[0][1][1]
    #         y2s=coordsr[0][0][1]
    #         x1s=coordsr[j][0][0]
    #         x2s=coordsr[j+1][0][0]
    #         # # %%
    #         slice = origs[y1s:y2s, x1s:x2s]
    #         # print(slice)

    #         gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
    #         blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    #         edged = cv2.Canny(blurred, 50, 100) # blurred

    #         contours, hierarchy= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         # print(contours)
    #         if len(contours)!=2:
    #             continue
    #         x=contours[0][0][0][0]
    #         y1c=contours[0][1][0][1]
    #         y2c=contours[1][1][0][1]
    #         dS = dist.euclidean((x, y1c), (x, y2c))
    #         # print(dS)

    #         dimS = dS / pixelsPerMetric
    #         # print(((dimS*2.54)*10))
    #         # print("{:.2f}mm".format((dimS*2.54)*10))
    #         dimSr.append(dimS)
    #     # # %%
    #     for k in range(n-1):
    #         # print("k=",k)

    #         ##### left
    #         origs = oorig.copy()
    #         y1s=coordsl[0][1][1]
    #         y2s=coordsl[0][0][1]
    #         x1s=coordsl[k+1][0][0]
    #         x2s=coordsl[k][0][0]
    #         # # %%
    #         slicek = origs[y1s:y2s, x1s:x2s]

    #         # print(slicek)
    #         gray = cv2.cvtColor(slicek, cv2.COLOR_BGR2GRAY)
    #         blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    #         edged = cv2.Canny(blurred, 50, 100) # blurred

    #         contours, hierarchy= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         # print(contours)
    #         if len(contours)!=2:
    #             continue
    #         x=contours[0][0][0][0]
    #         y1c=contours[0][1][0][1]
    #         y2c=contours[1][1][0][1]
    #         dS = dist.euclidean((x, y1c), (x, y2c))
    #         # print(dS)

    #         dimS = dS / pixelsPerMetric
    #         # print(((dimS*2.54)*10))
    #         # print("{:.2f}mm".format((dimS*2.54)*10))
    #         dimSl.append(dimS)
    #     kedian = statistics.median(dimSl+dimSr)
    #     return kedian

    # kmls = []
    # for l in range(len(pu)):
    #     temp = bcd_lr(pu[l], pd[l])
    #     # print(pu[l], pd[l])
    #     kmls.append(temp)
###############################

    # for t in tc:
    #     approxac=0
    #     for approxa in abcde:
    #         for popints in range(1,len(approxa)//2):
    #             # print(approxa[popints], approxa[-popints])
    #             # print(popints)
    #             # print("len(approxa//2:",len(approxa)//2)
    #             # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    #             dA = dist.euclidean(rabcde[approxac][popints][0], rabcde[approxac][-popints][0])
    #             dimA = dA / pixelsPerMetric

    #             cv2.line(orig, approxa[popints][0]+[x1,y1], approxa[-popints][0]+[x1,y1], (0, 255, 0), 1)
    #             cv2.circle(orig, approxa[popints][0]+[x1,y1], 3, (0, 255, 0), -1)
    #             cv2.circle(orig, approxa[-popints][0]+[x1,y1], 3, (0, 255, 0), -1)
    #             # if popints!=1:
    #                 # continue
    #             cv2.putText(orig, "{:.2f}mm".format((dimA*2.54)*10+4),
    #                         # (int(tl[0])-15, int(tl[1])-15), 
    #                         approxa[popints][0]+[x1,y1], 
    #                         cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.35, (0, 255, 0), 1)
    #         approxac+=1
    #     green=orig.copy()
    


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            cabcde = process_file(os.path.join(UPLOAD_FOLDER, filename), filename)

            # filelist = filelist
            imageList = os.listdir('static/downloads/processed')
            for val in imageList:
                # if 'caio' in val:
                if re.search('aio', val):
                    # imageList.remove(imageList.index(val))
                    imageList.remove    (val)
                    
            # imagelist = ['processed/' + image for image in imageList]
            imagelist = imageList

            data={
                # "processed_img":'static/downloads/processed'+filename,
                "uploaded_img":'static/uploads/'+filename,
                
                "gray_img":'static/downloads/gray_'+filename,
                "blurred_img":'static/downloads/blurred_'+filename,
                "edged_img":'static/downloads/edged_'+filename,
                "dilated_img":'static/downloads/dilated_'+filename,
                "eroded_img":'static/downloads/eroded_'+filename,
                "thresh_img":'static/downloads/thresh_'+filename,
                "mask_img":'static/downloads/mask_'+filename,
                "tilesdetected":'static/downloads/tilesdetected_'+filename,
                "caio":'static/downloads/processed/caio_'+filename,
                # "processed_img_c{i}":'static/downloads/processed_'+'c{i}_'+filename,
                # "processed_img_c{i}":'static/downloads/processed_'+'c{i}_'+filename,
                "{i}":'static/downloads/processed_'+'c{i}_'+filename,
                # "counter":10 # doesnt work
                "greendotsi":'static/downloads/processed/greendotsi_'+filename,
                "cabcde":cabcde
                }

            return render_template("index.html",data=data, imagelist=imagelist)  
    return render_template('index.html')

    
  
# download 
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

# if __name__ == '__main__':
#     print(__name__)
    # app.run(port=5002)

# from flask import Flask
# app = Flask(__name__)
# # app.run(host='0.0.0.0', port=8080,debug=True)
