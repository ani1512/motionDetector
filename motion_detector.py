import cv2, pandas
import os
from datetime import datetime

first_frame = None
status_list = [None,None]
times = []
df = pandas.DataFrame(columns = ["Start","End"])

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

video = cv2.VideoCapture(0)
currentFrame = 0


while True:

    check, frame = video.read() #starts capturing the video frame by frame

    status = 0 #flag that detects motion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #converts color to gray
    gray = cv2.GaussianBlur(gray,(21,21),0)  #blurs the image which increases accuracy while processing

    if first_frame is None:
        first_frame = gray  #once the first frame has the numpy array, it will remain static throughout
        continue

    delta_frame = cv2.absdiff(first_frame, gray) # find the difference between first frame and gray frame

    """if the difference in a pixel of the first_frame and the gray frame (delta_frame) is
    >30 then assign the pixel a white colour. if the difference is <30, then
    assign the pixel a black colour. Thus we will get the threshold frame. The inbuilt
    method which does this is the threshold method"""

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    #frame, limit, colour (255 for white), method used to calculate threshold
    #threshold method returns a tuple of 2 elements.
    #When using THRESH_BINARY for calculating threshold,
    #we should access the 2nd element, when using any other method, we should access
    #the first element

    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 5)
    #Smoothens the thresh frame

    (_, cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html

    for contour in cnts:

        if cv2.contourArea(contour) < 10000:
            continue
        status = 1 #an object entered the frame

        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    status_list.append(status)  #appends the current status of the object

    if status_list[-1] == 1 and status_list[-2] == 0:   #checks for the exit of an object
            times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:   #checks for the entry of an object
            times.append(datetime.now())
            name = './data/frame' + str(currentFrame) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1



    cv2.imshow("Capturing",frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status_list == 1:
            times.append(datetime.now())
        break;


for i in range(0, len(times)-1, 2):
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index = True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
