import cv2
import numpy as np

im = cv2.imread('hole.jpg')

gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray=cv2.threshold(gray,20,255,cv2.THRESH_BINARY)[1]
cv2.imshow('gray',gray)

contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE   )

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area<400:
        a=cv2.drawContours(im,[cnt],2,(255,0,0),7)

cv2.imshow('im',a)
cv2.waitKey()
