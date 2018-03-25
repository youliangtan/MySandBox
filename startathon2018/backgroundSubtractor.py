import numpy as np
import cv2
import tracking
import csv

# ======================== ini param ==========================
coorList = []
areaThresh = 200
frame = 0
g1 = c1 = g2 = c2 = x1= x2= x3= x4 = 0
borderThresh = 0.08 #percent
incount_Thresh = 4
outcount_Tresh = 6
cross_timeThresh = 250 #milliseconds
lastOut_Detection = 0
distance = 10
skipFrame = 3
currentFrame = 0
trackerList = []
completedList =[]
neighbour_maxRadius = 40 # for point tracking



# ========================= end param =========================

"""
tracker list>>
[{inTime: ,outTime: ,speed: ,size: , lane:  }, {inTime: ,outTime: ,speed: ,size: , lane:  }, .....]
dict within a list

point tracking:
empty list, []
when create a 'in' point,  [[inTime, pX, pY , Incount, outcount ]], then keep on update this list, when out, remove this point,
when new 'in' point is detected before the out, [[inTime ,pX, pY, Incount, outcount ], [inTime , pX, pY, Incount, outcount ]] 
        another in point is created
when out point detected, point being removed, update dict

"""

def output_csv():
    ini_csv = ['outTime', 'inTime', 'Speed', 'Size','Lane']
    path = './output.csv'

    with open(path, 'w') as csvfile:
        wr = csv.writer(csvfile, quoting = csv.QUOTE_ALL )
        wr.writerow(ini_csv)
        for ele in completedList:
            wr.writerow(ele)

    print("File has successfully outputed") 

def contours_filtering(contours, hierarchy):
    newContours = []
    if len(contours) != 0:
        for cnt, h in zip(contours, hierarchy[0]):
            area = cv2.contourArea(cnt)
            #area filtering
            #add contour larger than certain size to new list
            if (area > areaThresh):
                # print(h)
                # print("hihi")
                if h[3] == -1: #make sure no parent in hiera
                    newContours.append(cnt)
    return newContours

# mouse callback function
def draw_boundary(event,x,y,flags,param):
    ##draw circle
    global g1, g2, c1, c2, x1, x2, x3, x4 #gradient and offset 
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("clicked")
        coorList.append([x,y])
        print("list: {} ".format(coorList))
        cv2.circle(frame,(x,y),5,(255,0,255),-1)
        
        #draw 2 lines
        if len(coorList) == 4:
            draw_line()
            g1 = float(coorList[1][1] - coorList[0][1])/(coorList[1][0]-coorList[0][0])
            c1 = coorList[1][1] - g1*coorList[1][0]

            g2 = float(coorList[3][1] - coorList[2][1])/(coorList[3][0]-coorList[2][0])
            c2 = coorList[2][1] - g2*coorList[2][0]
            print("g1 {}, c1 {}, g2 {}, c2 {}".format(g1, c1, g2, c2))
            
            if coorList[0][0] > coorList[1][0]:  # to let x1 smaller than x2
                x1 = coorList[1][0]
                x2 = coorList[0][0]
            else:
                x1 = coorList[0][0]
                x2 = coorList[1][0]

            if coorList[2][0] > coorList[3][0]:  # to let x3 smaller than x4
                x3 = coorList[3][0]
                x4 = coorList[2][0]
            else:
                x3 = coorList[2][0]
                x4 = coorList[3][0]

            print("x1, x2 range {} {}, x3, x4 range {} {}".format(x1, x2, x3, x4))


def draw_line():
    if coorList != []: #to check whether coorList is not empty
        if (len(coorList) >= 2):
            cv2.line(frame,(coorList[0][0],coorList[0][1]),(coorList[1][0],coorList[1][1]),(255,0,0),5)
        if (len(coorList) >= 4):
            cv2.line(frame,(coorList[2][0],coorList[2][1]),(coorList[3][0],coorList[3][1]),(255,0,0),5)

def ini():
    global frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (853, 480), interpolation = cv2.INTER_LINEAR) 

    while (1):
        if cv2.waitKey(33) == ord(' '):
            print "pressed space"
            break
        draw_line()
        cv2.imshow('image', frame)


    ''' enter x,y coor of veh point to function, return true: crossed line; false: didnt cross'''
def crossIn_Line(x_, y_):
    new_c = y_ - g1*x_
    if ( c1*(1-borderThresh) < new_c < c1 *(1+borderThresh)): ## check within c offset
        if ( x1 < x_ < x2): #check is it within the line range ()
            return True
            # print("crossed line, in count {}, time {}".format(inline_Count, lastIn_Detection))
    return False

# in count need change to new convention
def crossOut_Line(x_, y_):
    new_c = y_ - g2*x_
    if ( c2*(1-borderThresh) < new_c < c2 *(1+borderThresh)): ## check within c offset
        if (x3 < x_ < x4): #check is it within the line range ()
            return True
            # print("crossed line, out count {}, time {}".format(outline_Count, outTime))
    return False


def completedTrack(x_, y_, cnt, outTime):
    print(">>>>>>>>>>>>>------------1 car crossed, out!! Time: {} ----------<<<<<<<<<<".format(outTime))
    
    #--------------------------------- output details to completedList --------------------------
    for pt in trackerList: #find inTime in trackerList
        if pt[1] == x_ and pt[2] == y_:
            inTime = pt[0]
            vel = calcSpeed( inTime , outTime)
            size = cv2.contourArea(cnt)
            lane = labelLane(x_)
            completedList.append([round(outTime/1000, 2), round(inTime/1000, 2), vel, size, lane])
            print("Done! {}".format([outTime, inTime, vel, size, lane]))
            return True
        else:
            inTime = -1 
    print("#Error: veh crossed but no inTime info in trackerList. x, y: {} {}".format(x_, y_))
    

def labelLane(x): #assume 3 lanes
    #x3 left out line, x4 right out line
    RoadLength = (x4 - x3)
    laneLength = RoadLength/3
    
    if (x3 <= x <= x3 + laneLength):
        return 1
    elif (x3 + laneLength < x <= x3 + 2*laneLength):
        return 2
    elif (x3 + 2*laneLength < x <= x4):
        return 3


def calcSpeed(inTime, outTime):
    velocity = -1
    if (inTime != outTime and inTime != -1):
        velocity = (3600/1000) *distance / ((outTime-inTime)/1000)
    print("Speed is {}".format(velocity) )
    return velocity

#invert img
def invert(imagem):
    imagem = (255-imagem)
    return imagem

def Trackbar_onChange(trackbarValue):
    global currentFrame, frame

    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    err,frame = cap.read()
    frame = cv2.resize(frame, (853, 480), interpolation = cv2.INTER_LINEAR) 
    # print("tb ",cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.imshow("image", frame)

def point_tracking(pX, pY):
    for idx in range( len(trackerList) ):
        # print(trackerList)
        tX = trackerList[idx][1]
        tY = trackerList[idx][2]
        distance_sq = (pX - tX)*(pX - tX) + (pY - tY)*(pY - tY) 
        # print("next point", pX, pY)
        # print("distance", distance_sq)
        if neighbour_maxRadius*neighbour_maxRadius > distance_sq:
            # print("===>update point {}: {} {}".format(idx, tX, tY))
            trackerList[idx][1] = pX
            trackerList[idx][2] = pY

#give a contour, return a left bottorm corner point index
def find_leftBottomCorner(ct):
    c1 = 3 #weightage
    c2 = 2
    pt_idx = 0
    maxSum = 0
    for idx in range(len(ct)):
        x = (853 - ct[idx][0][0])
        y = ct[idx][0][1]
        sum = c1*x + c2*y
        if maxSum < sum:
            maxSum = sum
            pt_idx = idx
    return pt_idx

# ======================================= Main starts here ==============================================

# cap = cv2.VideoCapture('dataset/video1.avi')
cap = cv2.VideoCapture('GP010471.MP4')

#unsuccessful change of fps
fps1 = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FPS, 10)
fps2 = cap.get(cv2.CAP_PROP_FPS)
print("~This is fps1,2:  {}, {}".format(fps1, fps2))


vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("~video length>> ", vid_length)

# ------------------------- ini of drawing features ------------------------------
#draw boundary
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_boundary) 
# create trackbars for color change
cv2.createTrackbar('start','image',0, vid_length, Trackbar_onChange)

##initialization
ini()
# ---------------------------- end ini -----------------------------

#background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=120, detectShadows=True)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

#trackbar status
start = 6200 #cv2.getTrackbarPos('start','image')
cap.set(cv2.CAP_PROP_POS_FRAMES, start)

while(1):
    ret, frame = cap.read()
    
    currentTime = cap.get(cv2.CAP_PROP_POS_MSEC)
    currentFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('start', 'image', currentFrame)

    if (frame != None):
        # ---------------- frame management ----------------
        frame = cv2.resize(frame, (853, 480), interpolation = cv2.INTER_LINEAR)   #resize frame
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        blurred = invert(blurred)
        fgmask = fgbg.apply(blurred)
        kernel = np.ones((5,5),np.uint8)
        # cv2.imshow('subtraction image', fgmask)

        dilation = cv2.dilate(fgmask,kernel,iterations = 2)
        erosion = cv2.erode(dilation,kernel,iterations = 2)
        
        contourImg = erosion
        
        #------------------Contour Detection ------------------
        _, contours, hierarchy = cv2.findContours(contourImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_filtering(contours, hierarchy)
        cv2.drawContours(frame, contours, -1, (255, 0, 255), 5)  



        #-------------------loop over each contour ----------------
        for c in contours:
            
            # ---------- recontour via polyDP contour -------------------
            # epsilon = 0.05*cv2.arcLength(c,True)
            # approx = cv2.approxPolyDP(c,epsilon,True)
            # # print(approx)
            # cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)


            #-------------------find reference point of contours ----------------
            # M = cv2.moments(c)
            # cX = int(M["m10"] / M["m00"]) #centerX
            # cY = int(M["m01"] / M["m00"]) #centerY

            idx = find_leftBottomCorner(c)
            cX = c[idx][0][0]
            cY = c[idx][0][1]
            cv2.circle(frame, (cX, cY), 7, (0, 255, 255), -1)

            # -------------------update tracking list for point tracking-----------------------
            if len(trackerList) != 0:
                point_tracking(cX, cY)

            # -------------------check centre point passing through in and out line ----------
            if (crossIn_Line(cX, cY) == True):
                print(trackerList)
                #check in trackerlist anot
                for pt, idx in zip(trackerList, range(99)):
                    
                    print("check: {} {}".format(pt[1], pt[2]))
                    
                    if pt[1] == cX and pt[2] == cY:
                        #it already existed
                        pt[3] =pt[3] + 1
                        if pt[3] == incount_Thresh:
                            print(">>>>>>>>>>>>>------------1 car crossed, in!! Time: {} ----------<<<<<<<<<<".format(currentTime))
                        break    
                else:
                    print("appending")
                    trackerList.append([currentTime, cX, cY, 0 , 0]) # append new list
            else:
                for pt, idx in zip(trackerList, range(99)):
                    if pt[1] == cX and pt[2] == cY and pt[3] < 4: #incount
                        #remove those who didnt reached incount
                        trackerList.pop(idx)
                        break

            # ------------ remove point from tracking list when outline, output speed ----------
            if (crossOut_Line(cX, cY) == True):
                for pt, idx in zip(trackerList, range(99)):
                    if pt[1] == cX and pt[2] == cY:
                        #it already existed
                        pt[4] =pt[4] + 1
                        if pt[4] > outcount_Tresh:
                            # ---------------- completion of 1 track, input to completed list ----------
                            completedTrack(cX, cY, c, currentTime)
                            
                            # ------------------remove point from list---------------------
                            print("removed pt in tracker list!!")
                            trackerList.pop(idx)
                            break 

        # ------------ Remove if points stays too long in tracker list --------------------
        if len(trackerList) != 0:
            if trackerList[0][0] < currentTime - 3000: #stays more than 3 seconds
                trackerList.pop(0)
            

        draw_line()
        cv2.imshow('erode + contouring', contourImg)
        cv2.imshow('image', frame)

        k = cv2.waitKey(30) & 0xff
        # -------------------------------- exit -----------------------------------
        if (k == 27 or currentFrame in range(vid_length - skipFrame, vid_length)):
            break
        
        #speed up via frame skipping
        # cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + skipFrame )


# -------------------------- output to csv file -----------------------------------
output_csv()


cap.release()
cv2.destroyAllWindows()