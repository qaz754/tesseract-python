import cv2
import numpy as np
#import imutils
import pytesseract
from PIL import Image

img = cv2.imread('file_0.png')

    # Convert to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
kernel = np.ones((2,2), np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)
#img = cv2.erode(img, kernel, iterations=1)
#for x in range(1, 5):
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
#img = imutils.skeletonize(img, size=(2, 2))
cv2.imwrite('ske3.png', img)

#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

#tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
a = pytesseract.image_to_string(Image.open('ske3.png'), lang='eng', config = tessdata_dir_config)
if 'Click the upside' and 'down picture'  in a:
    print('success')
    #print('Click the upside down picture')
    print(a)
elif 'You already visited this advertisement in the last 24 hours' in a :
    print('error')
    print(a)
elif 'Thanks' in a:
    print('done')
    print(a)
else:
    print('error')
    print(a)
