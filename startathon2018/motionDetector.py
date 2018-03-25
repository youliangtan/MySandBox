
import numpy as np
import cv2
import math
import time
import matplotlib.pyplot as plt
from pygame import mixer
# from flask import Flask, render_template, request, jsonify, send_from_directory


param1 =20
sizeTresh = 80
pointNumTresh = 10
distanceTresh = 60
centroidsList = []

flask_status = 0
X_centriodsList = []
Y_centriodsList = []
all_centriodsList = []
centroid_ini_elements = 200
c_Y = 0



facePath = "haarcascade_upperbody.xml"
fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=20, detectShadows=True)
kernel = np.ones((5,5),np.uint8)



#trackbar 1 change event function
def Trackbar_onChange1(trackbarValue):
    global param1
    param1 = trackbarValue
    return 0


def find_if_close(x1, y1,x2, y2):
    dist = math.hypot(x1 - x2, y1 - y2)    
    if dist < distanceTresh:
        return True
    return False

def ini():
    #make it global var
    global  width, height, cap, all_centriodsList
    
    #webcam
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    height = int(cap.get(4))

    for idx in range(0, centroid_ini_elements):
        all_centriodsList.append([width/2, height/2])

def invert(imagem):
    imagem = (255-imagem)
    return imagem


def main():
    global param1, c_Y
    cv2.namedWindow('img')
    cv2.createTrackbar('param1','img', 1, 300, Trackbar_onChange1)
    cv2.setTrackbarPos('param1','img', param1)
    mixer.init()


    while (True):
        #capture from webcam
        centroidsList = []

        # all_centriodsList = []

        ret, frame = cap.read()
        frame = cv2.resize(frame,(0,0),fx=1,fy=1)
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)

        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        img = np.zeros((height, width,3), np.uint8)
        img = blurred
        thresh = np.zeros((height, width,3), np.uint8)

        # face_cascade = cv2.CascadeClassifier(facePath)
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(100,100))
        # for (x, y, w, h) in faces:
        #     print "1: there is here"
        #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # cv2.imshow("Output image", frame)
        img = invert(img)

        fgmask = fgbg.apply(blurred)
        erosion = cv2.erode(fgmask,kernel,iterations = 3)
        dilation = cv2.dilate(erosion,kernel,iterations = 2)

        _, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 255), 5)  
        for cnt in contours:
            size = cv2.contourArea(cnt)

            # size filtering 
            if size > sizeTresh:
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(img,(cX, cY), 5, (0,255,0), -1)
                all_centriodsList.append([cX, cY])
                all_centriodsList.pop(0)


# -------------------- failed attempted 2
        # # print all_centriodsLisst
        centroids_numpy = np.array([all_centriodsList])
        # print centroids_numpy
        
        M2 = cv2.moments(centroids_numpy)
        if int(M2["m00"]) != 0:

            c_X = int(M2["m10"] / M2["m00"])
            c_Y = int(M2["m01"] / M2["m00"])
            cv2.circle(img,(c_X, c_Y), 20, (255,255,255), -1)
        # all_centriodsList.append([cX, cY])
    

        


            #     if len(centroidsList) == 0:
            #         centroidsList.append([])
                
            #     for lst in centroidsList:
            #         if len(lst) == 0:
            #             lst.append([cX, cY])
            #             break
            #         elif len(lst) == 1:
            #             dist_boolean = find_if_close(lst[0][0], lst[0][1], cX, cY)
            #             if dist_boolean is True:
            #                 lst.append([cX, cY])
            #             else:
            #                 centroidsList.append([cX, cY])
            #                 break
            #         else:
            #             print lst[0][0]
            #             dist_boolean1 = find_if_close(lst[1][0], lst[1][1], cX, cY)
            #             dist_boolean2 = find_if_close(lst[0][0], lst[0][1], cX, cY)
            #             if dist_boolean1 is True:
            #                 lst[1][0] = cX 
            #                 lst[1][1] = cY
            #             elif dist_boolean2 is True:
            #                 lst[0][0] = cX 
            #                 lst[0][1] = cY
            #             else:
            #                 centroidsList.append([cX, cY])
            #                 break

            # print centroidsList
            # if len(centroidsList) != 0:
            #     if len(centroidsList[0]) == 2:
            #         cv2.circle(img,(int (centroidsList[0][0]), int(centroidsList[0][1])), 10, (255,255,255), -1)
            #         cv2.circle(img,int(centroidsList[1][0]), int(centroidsList[1][1]), 10, (255,255,255), -1)
                            


        cv2.imshow("contours", img)
        cv2.imshow("Final image", erosion)


        ch = cv2.waitKey(1)

        if ch & 0xFF == ord('q'):
            break

                # ----------------------- combine and filter ------------------
        
        if param1 != 20:
            c_Y = 999
            hitCount = 999


        if c_Y > 350:
            hitCount = hitCount + 1
            print "count is {}".format(hitCount)
            if hitCount > 80:
                print "____________________ hit ______________"
                mixer.music.load('./speech.mp3')
                mixer.music.play()
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(4)
                mixer.music.load('./speech.mp3')
                mixer.music.play()
                while(1):
                    print "wait "
                    time.sleep(4)
                    mixer.music.load('./ringing.mp3')
                    mixer.music.play()

        else:
            hitCount = 0



    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        plt.pause(0.005)
        ini()
        print("width: {}, height: {}".format(width, height))
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)