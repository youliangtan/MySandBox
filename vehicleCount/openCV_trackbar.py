import cv2

skipFrame = 5

# cap = cv2.VideoCapture('dataset/video1.avi')
cap = cv2.VideoCapture('GP010471.MP4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    err,img = cap.read()
    cv2.imshow("mywindow", img)
    pass

cv2.namedWindow('mywindow')
cv2.createTrackbar( 'start', 'mywindow', 0, length, onChange )
cv2.createTrackbar( 'end'  , 'mywindow', 100, length, onChange )

onChange(0)
cv2.waitKey()

start = cv2.getTrackbarPos('start','mywindow')
end   = cv2.getTrackbarPos('end','mywindow')
if start >= end:
    raise Exception("start must be less than end")

cap.set(cv2.CAP_PROP_POS_FRAMES,start) # change the frame to be captured nxt

while cap.isOpened():
    err,img = cap.read()

    currentFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(currentFrame)

    if currentFrame >= end:
        print("reached end of frame")
        break
    cv2.imshow("mywindow", img)
    k = cv2.waitKey(10) & 0xff
    if k==27:
        break
    
    #speed up
    cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame + skipFrame )