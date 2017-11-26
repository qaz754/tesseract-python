import cv2
import numpy as np
from PIL import Image
from cv2 import boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, imread, morphologyEx, pyrDown, rectangle, threshold
import pyautogui
import random

pyautogui.screenshot('Screenshot.png')
image = cv2.imread('Screenshot.png')
#image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#lower_red = np.array([7,150,100])
#upper_red = np.array([27,255,255]) # cam cliquesteria
#lower_tim = np.array([138,100,100])
#upper_tim = np.array([158,255,255]) # tim neobux

#lower_cam = np.array([7,150,100])
#upper_cam = np.array([27,255,255]) # cam neobux

#lower_cam = np.array([11,100,100])
#upper_cam = np.array([31,255,255]) # cam eldibux

lower_luc = np.array([49, 218, 195])
upper_luc = np.array([81, 255, 255])  # luc cliquebook

#lower_luc = np.array([50,255,100])
#upper_luc = np.array([70,255,255]) # luc

#lower_xanh = np.array([21,227,221])
#upper_xanh = np.array([109,255,255])  # xanh difbux

#lower_cam = np.array([9, 255, 247])
#upper_cam = np.array([29, 255, 255])  # cam difbux

#mask_tim = cv2.inRange(hsv, lower_tim, upper_tim)
#mask_cam = cv2.inRange(hsv, lower_cam, upper_cam)
#mask_xanh = cv2.inRange(hsv, lower_xanh, upper_xanh)
mask_luc = cv2.inRange(hsv, lower_luc, upper_luc)

#mask = mask_xanh
mask = mask_luc
cv2.imwrite('mask.png',mask)
#cv2.imwrite('res.png',res)

################################
image = cv2.imread('mask.png')
template = cv2.imread('mask_eldibux.png')

# resize images
#image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
#template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)

# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Find template
result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
if max_val > 0.4 :
    print (max_val)
    top_left = max_loc
    h,w = templateGray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    print ('a',top_left)
    print ('b',bottom_right)
    #pyautogui.click((top_left[0]+bottom_right[0])/2, (top_left[1]+bottom_right[1])/2)
    cv2.rectangle(image,top_left, bottom_right,(0,0,255),2)
    cv2.imwrite('mask1.png',image)

    #pyautogui.screenshot('2.png')
    rgb = imread('mask1.png')
    #rgb = imread('1.png')
    # downsample and use it for processing
    #rgb = pyrDown(large)
    # apply grayscale
    small = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # morphological gradient
    morph_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))# 1 4
    grad = morphologyEx(small, cv2.MORPH_GRADIENT, morph_kernel)
    # binarize
    _, bw = threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morph_kernel = getStructuringElement(cv2.MORPH_RECT, (3, 1)) #9 1
    # connect horizontally oriented regions
    connected = morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
    mask = np.zeros(bw.shape, np.uint8)
    # find contours
    im2, contours, hierarchy = findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours

    for idx in range(0, len(hierarchy[0])):
        rect = x, y, rect_width, rect_height = boundingRect(contours[idx])
        # fill the contour
        mask = drawContours(mask, contours, idx, (255, 255, 2555), cv2.FILLED)
        # ratio of non-zero pixels in the filled region
        r = float(countNonZero(mask)) / (rect_width * rect_height)
        if r > 0.55 and rect_height > 8 and rect_width > 4 and top_left[0] <= x and top_left[1] <= y and (top_left[0] + w) >= (x+rect_width) and (top_left[1] + h) >= (y+rect_height):
            rgb = rectangle(rgb, (x,y), (x+rect_width, y+rect_height), (0,255,0),1)
            print ('x',x)
            print ('y',y)
            print ('y+rect_height',y+rect_height)
            print ('x+rect_height',x+rect_width)
            print (rect_width)
            print ('------')
            x = random.randrange(x, x+rect_width-1)
            y = random.randrange(y, y+rect_height-1)
            pyautogui.moveTo(x, y, duration=0.15)
            #pyautogui.click(x , y)


#if top_left[0] < x and top_left[1] < y and (top_left[0] + w) > (x+rect_width) and (top_left[1] + h) > (y+rect_height) :
    cv2.imshow('captcha_result', rgb)
    cv2.waitKey()
