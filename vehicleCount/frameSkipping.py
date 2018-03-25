import numpy as np
import cv2

#################### Setting up the file ################
videoFile = 'GP010471.MP4'
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()

#################### Setting up parameters ################

#OpenCV is notorious for not being able to good to 
# predict how many frames are in a video. The point here is just to 
# populate the "desired_frames" list for all the individual frames
# you'd like to capture. 

fps = vidcap.get(cv2.CAP_PROP_FPS)
est_video_length_minutes = 3         # Round up if not sure.
est_tot_frames = est_video_length_minutes * 60 * fps  # Sets an upper bound # of frames in video clip

n = 5                             # Desired interval of frames to include
desired_frames = n * np.arange(est_tot_frames) 


#################### Initiate Process ################

for i in desired_frames:
    vidcap.set(1,i-1)                      
    success,image = vidcap.read(1)         # image is an array of array of [R,G,B] values
    frameId = vidcap.get(1)                # The 0th frame is often a throw-away
    # print("creating")
    cv2.imshow('image', image)
    # cv2.imwrite("FolderFrames/frame%d.jpg" % frameId, image)

vidcap.release()
print "Complete"