
import numpy as np

from numpy import *

from numpy import linalg as la

from PIL import Image

import glob

from matplotlib import pyplot as plt

def loadImageSet(add):

 filenames = glob.glob('face/pgm/*.pgm')

 filenames.sort()

 img = [Image.open(fn).convert('L').resize((98, 116)) for fn in filenames]

 FaceMat = np.asarray([np.array(im).flatten() for im in img])

 return FaceMat

def ReconginitionVector(selecthr = 0.8):

 # step1: load the face image data ,get the matrix consists of all image

 FaceMat = loadImageSet('face/yalefaces/')

 print('-----------FaceMat.shape--------')

 print(FaceMat.shape)

 # step2: average the FaceMat

 avgImg = mean(FaceMat,0)

 # step3: calculate the difference of avgimg and all image data(FaceMat)

 diffTrain = FaceMat-avgImg

 covMat =np.asmatrix(diffTrain) * np.asmatrix(diffTrain.T)

 eigvals,eigVects = linalg.eig(covMat) #la.linalg.eig(np.mat(covMat))

 #step4: calculate eigenvector of covariance matrix (because covariance matrix will cause memory error)

 eigSortIndex = argsort(-eigvals)

 for i in range(shape(FaceMat)[1]):

 if (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= selecthr:

 eigSortIndex = eigSortIndex[:i]

 break

 covVects = diffTrain.T * eigVects[:,eigSortIndex] # covVects is the eigenvector of covariance matrix

 # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵

 return avgImg,covVects,diffTrain

def judgeFace(judgeImg,FaceVector,avgImg,diffTrain):

 diff = judgeImg - avgImg

 weiVec = FaceVector.T* diff.T

 res = 0

 resVal = inf

#==============================================================================
# plt.imshow(avgImg.reshape(98,116))
# plt.show()
#==============================================================================

 for i in range(15):

 TrainVec = (diffTrain[i]*FaceVector).T

 if (array(weiVec-TrainVec)**2).sum() < resVal:
    res = i

 resVal = (array(weiVec-TrainVec)**2).sum()

 return res+1

if __name__ == '__main__':

 avgImg,FaceVector,diffTrain = ReconginitionVector(selecthr = 0.8)

 nameList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']

 characteristic = ['centerlight','glasses','happy','leftlight','noglasses','rightlight','sad','sleepy','surprised','wink']

 for c in characteristic:

     count = 0

 for i in range(len(nameList)):
 # 这里的loadname就是我们要识别的未知人脸图，我们通过15张未知人脸找出的对应训练人脸进行对比来求出正确率
 loadname = 'face/yalefaces/subject'+nameList[i]+'.'+c+'.pgm'
 judgeImg = Image.open(loadname).convert('L').resize((98, 116))
 #print(loadname)
 if judgeFace(mat(judgeImg).flatten(),FaceVector,avgImg,diffTrain) == int(nameList[i]):
     count += 1
print('accuracy of %s is %f'%(c, float(count)/len(nameList))) # 求出正确率