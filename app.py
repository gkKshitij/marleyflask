from itertools import count
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,flash 
from werkzeug.utils import secure_filename
import cv2
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import re

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
    detect_object(path, filename)
    
###
def midpoint(ptA, ptB): # to use in below function
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

filelist = []
def detect_object(path, filename):    # TODO:
    
    image = cv2.imread(path)
    
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

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = 60 # hard coded value

    orig = oorig.copy()
    tc = []

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
        print("Area of contour ", cv2.contourArea(c))
        if cv2.contourArea(c) < 3000  or cv2.contourArea(c) > 5200:
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

        print(nooftiles)
        nooftiles+=1
        
        cv2.putText(orig, f"{nooftiles}",
                    (int(tl[0])+15, int(tl[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2)

        cv2.imshow("Imager", orig)
        print("Area of contour ", cv2.contourArea(c))

        if cv2.contourArea(c) < 5200:
            tc.append(box)
            #continue
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"{DOWNLOAD_FOLDER}tilesdetected_{filename}",orig)


    counter = 0

    for t in tc:
        # print(c)
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
            print(contournp)
            filelist.append(contournp)
            print(counter)

        # cv2.destroyAllWindows()

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
            process_file(os.path.join(UPLOAD_FOLDER, filename), filename)

            # filelist = filelist
            imageList = os.listdir('static/downloads/processed')
            for val in imageList:
                # if 'caio' in val:
                if re.search('aio', val):
                    # imageList.remove(imageList.index(val))
                    imageList.remove(val)
                    
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
                }
            return render_template("index.html",data=data, imagelist=imagelist)  
    return render_template('index.html')

    
  
# download 
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
