import cv2
import numpy as np
from PIL import Image
from cv2 import boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, imread, morphologyEx, pyrDown, rectangle, threshold
import pyautogui


#pyautogui.screenshot('Screenshot.png')
rgb = imread('source3.png')
#rgb = imread('1.png')
# downsample and use it for processing
#rgb = pyrDown(large)
# apply grayscale
small = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# morphological gradient
morph_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (1, 25))# 1 25 dai
grad = morphologyEx(small, cv2.MORPH_GRADIENT, morph_kernel)
# binarize
_, bw = threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
morph_kernel = getStructuringElement(cv2.MORPH_RECT, (1, 1)) #9 14 cao
# connect horizontally oriented regions
connected = morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
mask = np.zeros(bw.shape, np.uint8)
# find contours
im2, contours, hierarchy = findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# filter contours
d = 0
for idx in range(0, len(hierarchy[0])):
    rect = x, y, rect_width, rect_height = boundingRect(contours[idx])
    # fill the contour
    mask = drawContours(mask, contours, idx, (255, 255, 2555), cv2.FILLED)
    # ratio of non-zero pixels in the filled region
    
    r = float(countNonZero(mask)) / (rect_width * rect_height)
    if r > 0.5 and rect_height > 50 and rect_width > 20 and rect_width < 100:
    #if r > 0.5 and rect_height >50 and rect_width > 20:#0.5 16 150
        rgb = rectangle(rgb, (x,y), (x+rect_width, y+rect_height), (0,255,0),1)
        print ('x',x)
        print ('y',y)
        print ('y+rect_height',y+rect_height)
        print ('x+rect_width',x+rect_width)
        #rgb=[].shape
        crop_img = rgb[y: y+rect_height, x: x+rect_width]
        filename = "file_%d.png"%d
        cv2.imwrite(filename, crop_img)
        d+=1
cv2.imshow('captcha_result', rgb)
cv2.imwrite('result.png', rgb)
#crop_img = rgb[71:154, 298:377] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
#cv2.imshow("cropped", crop_img)



cv2.waitKey()
