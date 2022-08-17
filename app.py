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

# arrangements for heroku
UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png','jpeg', 'JPG', 'PNG'}
app = Flask(__name__, static_url_path="/static")


# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'opencv'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 6mb
app.config['MAX_CONTENT_LENGTH'] = 7 * 1024 * 1024


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
                "processed_img":'static/downloads/processed_'+filename
                }
            return render_template("index.html",data=data)  
    return render_template('index.html')


def process_file(path, filename):
    detect_object(path, filename)
    
###
def midpoint(ptA, ptB): # to use in below function
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def detect_object(path, filename):    # TODO:
    
    image = cv2.imread(path)
    
    # image = cv2.resize(image,(480,360))
    # (h, w) = image.shape[:2]
    image = imutils.resize(image, height = 500) # resize
    oorig = image.copy()

    
################################
##
##

    y1=327 #a
    y2=385 #b
    x1=218 #c
    x2=491 #d

    roi = image[y1:y2, x1:x2]
    image = roi.copy()

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
    for c in cnts:
        
        
        if cv2.contourArea(c) < 100:
            continue
        
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
    
        box = perspective.order_points(box)
        box = box.astype("int")+[x1,y1]
    
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
    
        processed = orig
        # show the output image
        # cv2.imshow("Imager", orig)
        # print("Area of contour ", cv2.contourArea(c))
    cv2.imwrite(f"{DOWNLOAD_FOLDER}processed_{filename}",processed)
    
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    
  
# download 
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
