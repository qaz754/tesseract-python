import cv2
import numpy as np
from matplotlib import pyplot as plt

#from pybuilder.core import use_plugin, init, Author, task

img = cv2.imread('Final_Picture.png')
#img = cv2.resize(img, (0,0), fx=3.5, fy=3.5)  
kernel1 = np.ones((5,5), np.uint8)
#img = cv2.dilate(img, kernel1, iterations=1)

kernel = np.ones((5,5),np.float32)/10
dst = cv2.filter2D(img,-1,kernel)

kernel = np.ones((7,7),np.float32)/30
dst1 = cv2.filter2D(img,-1,kernel)

#blur = cv2.blur(img,(5,5))

#blur1 = cv2.GaussianBlur(img,(5,5),3)

median = cv2.medianBlur(dst1,5)

final = cv2.medianBlur(np.float32(img)/2,1)

closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)

#dst1 = cv2.fastNlMeansDenoisingColored(median,None,10,10,7,21)

cv2.imshow('source', img) #Show the image
cv2.imshow('dst', dst) #Show the image
cv2.imwrite('dst.png', dst) #Show the image
#cv2.imshow('blur', blur) #Show the image
#cv2.imshow('blur1', blur1) #Show the image
cv2.imshow('median', median) #Show the image
cv2.imshow('final', final) #Show the image
cv2.imshow('closing', closing) #Show the image
cv2.imshow('dst1', dst1) #Show the image
cv2.imwrite('median.png', dst1)

cv2.waitKey()

