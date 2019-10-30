import cv2
import numpy as np
import os
import math

try:
    if not os.path.exists('data2'):
        os.makedirs('data2')
except OSError:
    print('error:Creating directory of data')


##recode every frame
# while(True):
#     ret,frame=cap.read()
#     name='./data/frame'+str(currentFrame)+'.jpg'
#     print('Creating...'+name)
#     cv2.imwrite(name,frame)
#     currentFrame+=1
# cap.release()
# cv2.destroyAllWindows()


#print frame every seconds

imagesFolder = "./data/fengzheng/video_pic"
for i in range(1,30):
    cap = cv2.VideoCapture("cut"+str(i)+".mp4")
    currentFrame = 0
    frameRate = cap.get(5)  # frame rate
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = imagesFolder + "/image"+str(i)+"_" + str(int(frameId)) + ".jpg"
            cv2.imwrite(filename, frame)
            print('Creating...' + filename)
    cap.release()


print ("done")