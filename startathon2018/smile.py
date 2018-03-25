import numpy as np
import cv2
import webbrowser
import time
import freenect


def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

img = cv2.imread("faces.jpeg",1)


def main():
    smilePath = "haarcascade_smile.xml"
    facePath = "haarcascade_frontalface_default.xml"
    startTime = time.time() - 20
    lastdetection = time.time()

    count = 1

    #capture from webcam
    cap = cv2.VideoCapture(0)

    while(True):

        #capture from webcam
        ret, frame = cap.read()

        #capture from kinect
        frame = get_video()

        frame = cv2.resize(frame,(0,0),fx=1,fy=1)
        # cv2.imshow("Frame",frame)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        face_cascade = cv2.CascadeClassifier(facePath)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(200,200))
        # print("height and width> {} {}".format(height, width))

        #smile is between 80-100

        for (x, y, w, h) in faces:
    	    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # print("xywh> {} {} {} {}".format(x,y,w,h))
            crop_img = gray[y+h/2:y+h, x:x+w] # Crop from x, y, w, h -> 100, 200, 300, 400
            #: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        
            smile_cascade = cv2.CascadeClassifier(smilePath)
            smile = smile_cascade.detectMultiScale(crop_img, scaleFactor=1.7, minNeighbors=22, minSize=(80,80))
        
            #detected smiles
            for (x2, y2, w2, h2) in smile:
    	        cv2.rectangle(crop_img, (x2,y2), (x2+w2,y2+h2), (255,0,0), 2)
                cv2.rectangle(frame, (x2 +x, y2 +y+h/2), (x2+w2 +x,y2+h2 + y+h/2), (255,0,0), 2)
                print("Smiled!!! count > " + str(count))
                count = count +1
        
                duration = time.time() - startTime
                lastdetection = time.time()
                print("Duration > " + str(duration))
            cv2.imshow("cropped", crop_img)
            cv2.moveWindow("cropped", 800,0)


    
        cv2.imshow("Image",frame)

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break
        
        if (count == 10 and duration > 20):
            print ("----- Reached threshold and reached duration, access link! ------")
            #access url
            webbrowser.open("http://10.27.68.170:8080/smile")
            
            #reinitialize
            startTime = time.time()
            count = 1
        
        #reinitialize when no 
        if ((time.time() - lastdetection) > 5):
            count = 1


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)