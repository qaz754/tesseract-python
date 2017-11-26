import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyautogui

list=[]
pyautogui.screenshot('my_screenshot.png',region=(0,0,1204,30))
img_rgb = cv2.imread('tab.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('tab1.png',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where( res >= threshold)
count = 0
for pt in zip(*loc[::-1]):
    tab = (pt[0], pt[1])
    list.append(tab)
    #pyautogui.click(pt[0] - 10, pt[1] )
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    count = count + 1
    #pyautogui.PAUSE = 1


print(list)
print(count)
cv2.imshow("Template", img_rgb)
cv2.waitKey(0)
#cv2.imwrite('res.png',img_rgb)
