import numpy as np
import cv2
import pyautogui

#pyautogui.screenshot('Screenshota.png',region=(0,92, 1353, 151))
#pyautogui.screenshot('Screenshota.png')
image = cv2.imread('source1.png')  # hinh Screenshot

template = cv2.imread('a.png')  # hinh mau

#image = cv2.resize(image, (0,0), fx=3, fy=3)
#template = cv2.resize(template, (0,0), fx=3, fy=3)
# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Find template
result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)# TM_CCORR_NORMED sd voi icon
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)              # TM_CCOEFF_NORMED sd voi anh
top_left = max_loc
h, w = templateGray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image,top_left, bottom_right,(0,0,255),1)
print('x',top_left)
print('y',bottom_right)
print(max_val)



# Show result
cv2.imshow("Template", template)
cv2.imshow("Result", image)

#cv2.imwrite('Result.png', result)

#cv2.moveWindow("Template", 10, 50);
#cv2.moveWindow("Result", 150, 50);

cv2.waitKey(0)
